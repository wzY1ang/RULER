import argparse, os, json
from pathlib import Path
import sys
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel

MODEL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_DIR))
from modeling_qwen3_embed import Qwen3ForEmbedding  # noqa: E402


# ---------- dist utils ----------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def is_main_process(rank): return rank == 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------- args ----------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="Columns: query,cand_text,group_id,cand_label[,group_label,cand_id]")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--q_max", type=int, default=256)
    ap.add_argument("--d_max", type=int, default=256)
    ap.add_argument("--batch_groups", type=int, default=8)
    ap.add_argument("--cutoff", type=int, default=10)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--debug_samples", type=int, default=5, help="Number of mixed-group rankings to print")
    ap.add_argument("--output_dir", default=None, help="Metrics output directory")
    ap.add_argument("--base_model_path", default=None, help="Base model path required for LoRA adapters")

    return ap.parse_args()

# ---------- data ----------
class GroupDataset(Dataset):
    def __init__(self, tsv):
        try:
            df = pd.read_csv(tsv, sep="\t",
                             usecols=["query","cand_text","group_id","cand_label","group_label","cand_id"],
                             dtype={"cand_label": int, "group_label": float, "cand_id": str}).fillna({"group_label": -1})
        except ValueError:
            df = pd.read_csv(tsv, sep="\t",
                             usecols=["query","cand_text","group_id","cand_label","group_label"],
                             dtype={"cand_label": int, "group_label": float}).fillna({"group_label": -1})
            df["cand_id"] = df.groupby("group_id").cumcount().astype(str)
            print("Warning: 'cand_id' is missing; using the row index instead.")

        self.groups, self.group_ids = [], []
        for gid, g in df.groupby("group_id", sort=False):
            q = g["query"].iloc[0]
            docs = g["cand_text"].tolist()
            y = torch.tensor(g["cand_label"].tolist(), dtype=torch.long)
            cand_ids = g["cand_id"].tolist()
            if (g["group_label"] >= 0).any():
                uniq = g["group_label"].dropna().unique()
                gl = float(uniq[0]) if len(uniq)==1 else (1.0 if (y==1).any().item() else 0.0)
            else:
                gl = 1.0 if (y==1).any().item() else 0.0
            self.groups.append((q, docs, y, gl, cand_ids))
            self.group_ids.append(gid)

    def __len__(self): return len(self.groups)
    def __getitem__(self, i):
        q, docs, y, gl, cand_ids = self.groups[i]
        gid = self.group_ids[i]
        return q, docs, y, gl, cand_ids, gid

def collate_groups(batch):
    Q, D, Y, lens, GL, CAND_IDS, GROUP_IDS = [], [], [], [], [], [], []
    for q, docs, y, gl, cand_ids, gid in batch:
        k = len(docs)
        Q += [q]*k
        D += docs
        Y.append(y)
        lens.append(k)
        GL.append(gl)
        CAND_IDS.extend(cand_ids)
        GROUP_IDS.extend([gid]*k)
    Y = torch.cat(Y, dim=0)
    GL = torch.tensor(GL, dtype=torch.float32)
    return Q, D, Y, lens, GL, CAND_IDS, GROUP_IDS

# ---------- encode ----------
def encode_pairs_split_trunc(tokenizer, queries, docs, *, q_max=256, d_max=256, device="cpu"):
    q_tok = tokenizer(queries, add_special_tokens=False, truncation=True, max_length=q_max, return_tensors=None)
    d_tok = tokenizer(docs,   add_special_tokens=False, truncation=True, max_length=d_max, return_tensors=None)
    q_ids_list = q_tok["input_ids"]; d_ids_list = d_tok["input_ids"]
    input_ids = []
    for q_ids, d_ids in zip(q_ids_list, d_ids_list):
        pair_ids = tokenizer.build_inputs_with_special_tokens(q_ids, d_ids)
        input_ids.append(pair_ids)
    pad_id = tokenizer.pad_token_id or 0
    Lmax = max(len(x) for x in input_ids)
    input_ids_pad = torch.full((len(input_ids), Lmax), pad_id, dtype=torch.long)
    attn_mask     = torch.zeros((len(input_ids), Lmax), dtype=torch.long)
    for i, ids in enumerate(input_ids):
        l = len(ids)
        input_ids_pad[i, :l] = torch.as_tensor(ids, dtype=torch.long)
        attn_mask[i, :l] = 1
    return {"input_ids": input_ids_pad.to(device), "attention_mask": attn_mask.to(device)}

# ---------- NDCG & MAP utils ----------
def dcg_at_k(scores, labels, k):
    order = torch.argsort(scores, descending=True)[:k]
    labels_sorted = labels[order]
    gains = 2**labels_sorted.float() - 1
    discounts = torch.log2(torch.arange(2, len(labels_sorted)+2, device=scores.device).float())
    return (gains / discounts).sum()

def ndcg_at_k(scores, labels, k):
    dcg = dcg_at_k(scores, labels, k)
    ideal_labels = torch.sort(labels, descending=True).values
    idcg = dcg_at_k(ideal_labels.float(), ideal_labels, k)
    return dcg / idcg if idcg > 0 else torch.tensor(0.0, device=scores.device)

def average_precision(scores, labels, k):
    order = torch.argsort(scores, descending=True)[:k]
    labels_sorted = labels[order]
    precisions, num_relevant = [], 0
    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    relevant_total = (labels == 1).sum().item()
    denom = max(1, min(relevant_total, k))
    return torch.tensor(sum(precisions) / denom, device=scores.device) if precisions else torch.tensor(0.0, device=scores.device)

# ---------- eval ----------
@torch.no_grad()
def evaluate_gpu(model, tok, loader, device, *, cutoff=10, T=1.0, q_max=256, d_max=256, amp=False, rank=0, world_size=1, save_per_group=False, args=None, out_dir=None):
    model.eval()
    RR = torch.zeros((), device=device)
    P5 = torch.zeros((), device=device)
    P10 = torch.zeros((), device=device)
    Hit5 = torch.zeros((), device=device)
    Hit10 = torch.zeros((), device=device)
    Recall5 = torch.zeros((), device=device)
    Recall10 = torch.zeros((), device=device)
    NDCG5 = torch.zeros((), device=device)
    NDCG10 = torch.zeros((), device=device)
    MAP10 = torch.zeros((), device=device)
    Top1Acc = torch.zeros((), device=device)
    pos_ct = torch.zeros((), device=device)

    NegKL = torch.zeros((), device=device)
    NegMax = torch.zeros((), device=device)
    neg_ct = torch.zeros((), device=device)

    total_gap = torch.zeros((), device=device)
    gap_ct = torch.zeros((), device=device)
    total_overlap = torch.zeros((), device=device)
    overlap_ct = torch.zeros((), device=device)
    total_pos_min = torch.zeros((), device=device)
    pos_min_ct = torch.zeros((), device=device)

    R_precision = torch.zeros((), device=device)
    FPR_at_05 = torch.zeros((), device=device)

    per_group_results = [] if save_per_group else None
    group_counter = 0


    nr_r_ratio = torch.zeros((), device=device)
    nr_r_ct = torch.zeros((), device=device)


    pbar = tqdm(loader, desc=f"Evaluating (Rank {rank})", unit="batch", disable=(rank != 0))

    for Q, D, Y, lens, GL, CAND_IDS, GROUP_IDS in pbar:
        enc = encode_pairs_split_trunc(tok, Q, D, q_max=q_max, d_max=d_max, device=device)
        with torch.amp.autocast('cuda', enabled=bool(amp)):
            out = model(enc["input_ids"], enc["attention_mask"].long(), mode="rerank")
            s = out["scores"].squeeze(-1)

        off = 0
        for L, gl in zip(lens, GL):
            ss = s[off:off+L].float()
            yy = Y[off:off+L].to(device)

            raw_ss = ss.float()


            ss = torch.sigmoid(ss)

            k = min(cutoff, L)


            if (yy == 0).all():
                max_val = ss.max()
                NegMax += max_val
                neg_ct += 1
                if max_val > 0.5:
                    FPR_at_05 += 1

            if (yy == 1).any():
                order = torch.argsort(ss, descending=True)
                top = yy[order[:k]]

                pos_pos = (top == 1).nonzero(as_tuple=True)[0]
                RR += (1.0 / (pos_pos[0].float() + 1.0)) if pos_pos.numel() > 0 else torch.zeros((), device=device)

                P5  += top[:min(5, k)].float().mean()
                P10 += top[:min(10, k)].float().mean()

                hit5  = (top[:min(5, k)]  == 1).any().float()
                hit10 = (top[:min(10, k)] == 1).any().float()
                Hit5  += hit5
                Hit10 += hit10

                pos_total = (yy == 1).sum().item()
                if pos_total > 0:
                    Recall5  += (top[:min(5, k)]  == 1).sum().item() / pos_total
                    Recall10 += (top[:min(10, k)] == 1).sum().item() / pos_total

                Top1Acc += (yy[order[0]] == 1).float()
                NDCG5   += ndcg_at_k(ss, yy, 5)
                NDCG10  += ndcg_at_k(ss, yy, 10)
                MAP10   += average_precision(ss, yy, 10)
                pos_ct  += 1

                pos_logits = raw_ss[yy == 1]
                neg_logits = raw_ss[yy == 0]

                if neg_logits.numel() > 0 and pos_logits.numel() > 0:

                    total_gap     += pos_logits.mean() - neg_logits.mean()
                    gap_ct        += 1


                    total_overlap += (neg_logits.max() >= pos_logits.min()).float()
                    overlap_ct    += 1


                    pos_min = pos_logits.min()
                    pos_max = pos_logits.max()


                    neg_in_relevant = ((neg_logits >= pos_min) & (neg_logits <= pos_max)).sum().float()

                    nr_r_ratio += neg_in_relevant / neg_logits.numel()
                    nr_r_ct += 1

                if pos_logits.numel() > 0:



                    total_pos_min += torch.sigmoid(pos_logits).min()
                    pos_min_ct    += 1

                pos_count = (yy == 1).sum().item()
                if pos_count > 0:
                    R_precision += (top[:pos_count] == 1).sum().item() / pos_count

                if save_per_group:
                    group_cand_ids = CAND_IDS[off:off+L]
                    group_gid = GROUP_IDS[off]
                    raw_scores = s[off:off+L].float().cpu().numpy().tolist()
                    norm_scores = ss.cpu().numpy().tolist()
                    labels = yy.cpu().numpy().tolist()
                    ord_np = order.cpu().numpy()
                    sorted_cids = [group_cand_ids[i] for i in ord_np]
                    sorted_raw = [raw_scores[i] for i in ord_np]
                    sorted_norm = [norm_scores[i] for i in ord_np]
                    sorted_labels = [labels[i] for i in ord_np]
                    top_labels = sorted_labels[:k]
                    rr_local = 1.0 / (top_labels.index(1) + 1) if 1 in top_labels else 0.0
                    p_at_10_local = sum(top_labels[:min(10,len(top_labels))]) / min(10,len(top_labels)) if len(top_labels)>0 else 0.0
                    per_group_results.append({
                        "group_id": group_gid,
                        "query": Q[off] if len(Q) > off else "N/A",
                        "num_candidates": L,
                        "has_positive": 1,
                        "MRR": rr_local,
                        "P@10": p_at_10_local,
                        "raw_scores": raw_scores,
                        "norm_scores": norm_scores,
                        "labels": labels,
                        "cand_ids": group_cand_ids,
                        "sorted_cand_ids": sorted_cids,
                        "sorted_raw_scores": sorted_raw,
                        "sorted_norm_scores": sorted_norm,
                        "sorted_labels": sorted_labels,
                    })
                    if is_main_process(rank) and group_counter < args.debug_samples:
                        print(f"\n=== Debug Sample {group_counter+1} (Group: {group_gid}) ===")
                        print(f"Query: {Q[off]}")
                        for idx, (cid, ns, rs, lab) in enumerate(zip(
                            sorted_cids[:10], sorted_norm[:10], sorted_raw[:10], sorted_labels[:10]
                        )):
                            mark = "POS" if lab == 1 else "NEG"
                            print(f"  {idx+1:2d}. {mark} {cid} | norm={ns:.4f} | raw={rs:.4f} | label={lab}")
                        group_counter += 1
            off += L

    if world_size > 1:
        for t in [RR, P5, P10, Hit5, Hit10, Recall5, Recall10, NDCG5, NDCG10, MAP10, Top1Acc, pos_ct,
                  NegKL, NegMax, neg_ct, total_gap, gap_ct, total_overlap, overlap_ct, total_pos_min, pos_min_ct,
                  R_precision, FPR_at_05, nr_r_ratio, nr_r_ct]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


    metrics = {}
    if pos_ct.item() > 0:
        denom = pos_ct
        metrics.update({
            "MRR@10": (RR/denom).item(),
            "P@5": (P5/denom).item(),
            "P@10": (P10/denom).item(),
            "Hit@5": (Hit5/denom).item(),
            "Hit@10": (Hit10/denom).item(),
            "Recall@5": (Recall5/denom).item(),
            "Recall@10": (Recall10/denom).item(),
            "NDCG@5": (NDCG5/denom).item(),
            "NDCG@10": (NDCG10/denom).item(),
            "MAP@10": (MAP10/denom).item(),
            "Top1_Accuracy": (Top1Acc/denom).item(),
            "PosGroups": int(denom.item()),
            "R-Precision": (R_precision/denom).item(),
        })
        if gap_ct.item() > 0:
            metrics["Avg_Score_Gap"] = (total_gap/gap_ct).item()
        if overlap_ct.item() > 0:
            metrics["Score_Overlap_Ratio"] = (total_overlap/overlap_ct).item()
        if pos_min_ct.item() > 0:
            metrics["Pos_MinScore(mean)"] = (total_pos_min/pos_min_ct).item()

        if nr_r_ct.item() > 0:
            metrics["NR_at_R"] = (nr_r_ratio / nr_r_ct).item()


    if neg_ct.item() > 0:
        metrics.update({
            "Neg_KL(mean)": (NegKL/neg_ct).item(),
            "Neg_MaxScore(mean)": (NegMax/neg_ct).item(),
            "NegGroups": int(neg_ct.item()),
            "FPR@0.5": (FPR_at_05/neg_ct).item(),
        })

    return metrics, per_group_results

from sklearn.metrics import roc_curve
import numpy as np

def calc_global_detailed_metrics(all_per_group):
    """
    Compute logit gap, calibrated gap, and optimal bias over the full dataset.
    """
    y_true = []
    y_logits = []


    for g in all_per_group:
        y_true.extend(g['labels'])
        y_logits.extend(g['raw_scores'])

    y_true = np.array(y_true)
    y_logits = np.array(y_logits)


    pos_logits = y_logits[y_true == 1]
    neg_logits = y_logits[y_true == 0]

    raw_logits_gap = 0.0
    if len(pos_logits) > 0 and len(neg_logits) > 0:
        raw_logits_gap = np.mean(pos_logits) - np.mean(neg_logits)


    fpr, tpr, thresholds = roc_curve(y_true, y_logits)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]


    optimal_bias = -best_threshold



    calibrated_probs = 1 / (1 + np.exp(-(y_logits + optimal_bias)))

    pos_probs_calib = calibrated_probs[y_true == 1]
    neg_probs_calib = calibrated_probs[y_true == 0]

    calibrated_gap = 0.0
    if len(pos_probs_calib) > 0 and len(neg_probs_calib) > 0:
        calibrated_gap = np.mean(pos_probs_calib) - np.mean(neg_probs_calib)

    return {
        "Raw_Logits_Gap": float(raw_logits_gap),
        "Optimal_Bias": float(optimal_bias),
        "Calibrated_Gap": float(calibrated_gap)
    }


# ---------- main ----------
def main():
    args = get_args()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    out_dir = args.output_dir or args.model_path
    if is_main_process(rank):
        os.makedirs(out_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()




    is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))

    if is_lora:
        if is_main_process(rank):
            print(f"Detected LoRA adapter: {args.model_path}")


        if args.base_model_path is None:
            raise ValueError("--base_model_path is required when --model_path contains a LoRA adapter")

        load_path = args.base_model_path
    else:
        if is_main_process(rank):
            print(f"Loading full model: {args.model_path}")
        load_path = args.model_path


    config = AutoConfig.from_pretrained(load_path)
    if not hasattr(config, 'parallel_attn') or config.parallel_attn is None:
        config.parallel_attn = False
    if not hasattr(config, 'attn_implementation') or config.attn_implementation is None:
        config.attn_implementation = "eager"

    tok = AutoTokenizer.from_pretrained(load_path)
    model = Qwen3ForEmbedding.from_pretrained(load_path, config=config).to(device)


    if is_lora:
        if is_main_process(rank):
            print("Merging LoRA adapter")
        model = PeftModel.from_pretrained(model, args.model_path)

    model.eval()

    ds = GroupDataset(args.tsv)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    ld = DataLoader(ds, batch_size=args.batch_groups, sampler=sampler, collate_fn=collate_groups)

    save_per_group = True

    metrics, per_group = evaluate_gpu(
        model, tok, ld, device,
        q_max=args.q_max, d_max=args.d_max,
        cutoff=args.cutoff, T=args.temp,
        rank=rank, world_size=world_size,
        save_per_group=save_per_group,
        args=args,
        out_dir=out_dir
    )

    part_file = None
    if per_group is not None:
        part_file = os.path.join(out_dir, f"per_group_rank{rank}.json")
        with open(part_file, "w", encoding="utf-8") as f:
            json.dump(per_group, f, indent=2, ensure_ascii=False)

    if world_size > 1:
        dist.barrier()

    all_per_group = None
    if is_main_process(rank):
        if world_size == 1:
            all_per_group = per_group
        else:
            all_per_group = []
            for r in range(world_size):
                part_path = os.path.join(out_dir, f"per_group_rank{r}.json")
                if os.path.exists(part_path):
                    with open(part_path, "r", encoding="utf-8") as f:
                        all_per_group.extend(json.load(f))

    if is_main_process(rank):
        print("="*60)
        print(f"Eval on {os.path.abspath(args.tsv)}")
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("="*60)


        metrics_json = os.path.join(out_dir, "metrics_final.json")
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics_final.csv"), index=False, encoding="utf-8")
        print(f"Final metrics saved to {metrics_json} and metrics_final.csv")

        if all_per_group:
            detailed_json = os.path.join(out_dir, "eval_detailed_per_group.json")
            with open(detailed_json, "w", encoding="utf-8") as f:
                json.dump(all_per_group, f, indent=2, ensure_ascii=False)
            print(f"Detailed per-group results saved to {detailed_json}")

            df_summary = pd.DataFrame([{
                "group_id": r["group_id"],
                "query": (r["query"][:100] + "...") if len(r["query"]) > 100 else r["query"],
                "num_candidates": r["num_candidates"],
                "has_positive": r["has_positive"],
                "MRR": r["MRR"],
                "P@10": r["P@10"],
            } for r in all_per_group])
            df_summary.to_csv(os.path.join(out_dir, "eval_summary_per_group.csv"), index=False, encoding="utf-8")
            print("Summary saved to eval_summary_per_group.csv")

            rows = []
            for r in all_per_group:
                for rank_idx, (cid, ns, rs, lab) in enumerate(zip(
                    r["sorted_cand_ids"], r["sorted_norm_scores"], r["sorted_raw_scores"], r["sorted_labels"]
                )):
                    rows.append({
                        "group_id": r["group_id"],
                        "rank": rank_idx+1,
                        "cand_id": cid,
                        "norm_score": ns,
                        "raw_score": rs,
                        "label": lab,
                        "is_top1": 1 if rank_idx==0 else 0,
                        "is_positive": 1 if lab==1 else 0,
                    })
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "eval_full_ranking.csv"), index=False, encoding="utf-8")
            print("Full ranking list saved to eval_full_ranking.csv")




    if is_main_process(rank):
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        if all_per_group:
            try:
                print("Computing global calibrated metrics...")
                global_metrics = calc_global_detailed_metrics(all_per_group)


                metrics.update(global_metrics)


                print(f"Raw_Logits_Gap = {global_metrics['Raw_Logits_Gap']:.4f}")
                print(f"Calibrated_Gap = {global_metrics['Calibrated_Gap']:.4f}")
                print(f"Optimal_Bias = {global_metrics['Optimal_Bias']:.4f}")

            except Exception as e:
                print(f"Could not calculate global metrics (is scikit-learn installed?): {e}")



        # print(f"Overlap Ratio: {metrics.get('Score_Overlap_Ratio', 'N/A')}")




        metrics_json = os.path.join(out_dir, "metrics_final.json")
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


        pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics_final.csv"), index=False, encoding="utf-8")
        print(f"Final metrics saved to {metrics_json} and metrics_final.csv")



    cleanup()

if __name__ == "__main__":
    main()
