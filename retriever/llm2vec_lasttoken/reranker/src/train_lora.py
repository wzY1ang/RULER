# train_reranker_group_label_ddp.py
# pip install torch transformers pandas matplotlib peft
import os, math, argparse, random, time, csv
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from qwen3forall import Qwen3ForEmbedding
import torch.distributed as dist
import io, json


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class TrainLogger:
    def __init__(self, out_dir, enable=True):
        self.enable = enable
        self.out_dir = out_dir
        self.step_csv = os.path.join(out_dir, "train_log.csv")
        self.epoch_csv = os.path.join(out_dir, "valid_log.csv")
        os.makedirs(out_dir, exist_ok=True)

        with open(self.step_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch","global_step","loss_avg","lr","grad_norm",
                "seq_len_avg","seq_len_max","pad_len",
                "avg_pos","avg_neg","avg_pairs",
                "step_sec","ema_step_sec","eta_sec","gpu_mem_alloc_gb","gpu_mem_reserved_gb"
            ])
        with open(self.epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "epoch","time_sec","MRR@10","P@5","P@10","PosGroups",
                "Neg_KL(mean)","Neg_MaxScore(mean)","NegGroups",
                "Top1_Acc","Min_Pos_Rank","Max_Neg_Rank","Rank_Gap","KL_Div"
            ])

    def log_step(self, **kw):
        if not self.enable: return
        with open(self.step_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                kw.get("epoch",0), kw.get("global_step",0), kw.get("loss_avg",0.0), kw.get("lr",0.0), kw.get("grad_norm",0.0),
                kw.get("seq_len_avg",0.0), kw.get("seq_len_max",0), kw.get("pad_len",0),
                kw.get("avg_pos",0.0), kw.get("avg_neg",0.0), kw.get("avg_pairs",0.0),
                kw.get("step_sec",0.0), kw.get("ema_step_sec",0.0), kw.get("eta_sec",0.0), kw.get("gpu_mem_alloc_gb",0.0), kw.get("gpu_mem_reserved_gb",0.0)
            ])

    def log_epoch(self, epoch:int, time_sec:float, metrics:dict):
        if not self.enable: return
        with open(self.epoch_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, time_sec,
                metrics.get("MRR@10", None), metrics.get("P@5", None), metrics.get("P@10", None), metrics.get("PosGroups", None),
                metrics.get("Neg_KL(mean)", None), metrics.get("Neg_MaxScore(mean)", None), metrics.get("NegGroups", None),
                metrics.get("Top1_Acc", None), metrics.get("Min_Pos_Rank", None), metrics.get("Max_Neg_Rank", None), metrics.get("Rank_Gap", None), metrics.get("KL_Div", None)
            ])

    def _safe_read_csv(self, path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def plot_curves(self):
        if not self.enable: return

        df = self._safe_read_csv(self.step_csv)
        if df is not None and len(df) > 0:
            try:
                # loss
                plt.figure()
                plt.plot(df["global_step"], df["loss_avg"])
                plt.xlabel("global_step"); plt.ylabel("loss(avg per log interval)")
                plt.title("Training Loss")
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "loss_curve.png"), dpi=150)
                plt.close()

                # lr
                plt.figure()
                plt.plot(df["global_step"], df["lr"])
                plt.xlabel("global_step"); plt.ylabel("learning_rate")
                plt.title("Learning Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, "lr_curve.png"), dpi=150)
                plt.close()

                # grad norm
                if "grad_norm" in df.columns:
                    plt.figure()
                    plt.plot(df["global_step"], df["grad_norm"])
                    plt.xlabel("global_step"); plt.ylabel("grad_norm")
                    plt.title("Gradient Norm")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, "grad_norm_curve.png"), dpi=150)
                    plt.close()
            except Exception:
                pass


        ev = self._safe_read_csv(self.epoch_csv)
        if ev is not None and len(ev) > 0:
            try:
                if "MRR@10" in ev.columns and ev["MRR@10"].notna().any():
                    plt.figure()
                    plt.plot(ev["epoch"], ev["MRR@10"])
                    plt.xlabel("epoch"); plt.ylabel("MRR@10")
                    plt.title("Validation MRR@10")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, "mrr_curve.png"), dpi=150)
                    plt.close()
                if "P@10" in ev.columns and ev["P@10"].notna().any():
                    plt.figure()
                    plt.plot(ev["epoch"], ev["P@5"].fillna(0))
                    plt.plot(ev["epoch"], ev["P@10"].fillna(0))
                    plt.xlabel("epoch"); plt.ylabel("precision")
                    plt.title("Validation Precision")
                    plt.legend(["P@5","P@10"])
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, "precision_curve.png"), dpi=150)
                    plt.close()
                if "Top1_Acc" in ev.columns and ev["Top1_Acc"].notna().any():
                    plt.figure()
                    plt.plot(ev["epoch"], ev["Top1_Acc"])
                    plt.xlabel("epoch"); plt.ylabel("Top1_Acc")
                    plt.title("Validation Top1 Accuracy")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, "top1_acc_curve.png"), dpi=150)
                    plt.close()
                if "KL_Div" in ev.columns and ev["KL_Div"].notna().any():
                    plt.figure()
                    plt.plot(ev["epoch"], ev["KL_Div"])
                    plt.xlabel("epoch"); plt.ylabel("KL_Div")
                    plt.title("Validation KL Divergence")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.out_dir, "kl_div_curve.png"), dpi=150)
                    plt.close()
            except Exception:
                pass

# ---------------- args ----------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True,
        help="Columns: query,cand_text,group_id,cand_label[,group_label]")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_groups", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--rank_loss", choices=["logistic","hinge"], default="logistic")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--beta_ent", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_pairs", type=int, default=1024)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_steps", type=int, default=5)
    ap.add_argument("--q_max", type=int, default=256)
    ap.add_argument("--d_max", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0, help="Dynamic margin: confidence scaling factor")
    ap.add_argument("--gamma", type=float, default=0.05, help="Dynamic margin: std compensation factor")
    ap.add_argument("--beta_IRDA", type=float, default=0.1, help="Ideal Ranking Distribution Alignment loss weight")


    ap.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension")
    ap.add_argument("--lora_alpha", type=float, default=64, help="LoRA scaling factor")
    ap.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    ap.add_argument("--target_modules", type=str, nargs='+',
                    default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    help="Target modules for LoRA adaptation")
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    ap.add_argument("--grad_checkpoint", action="store_true", help="Enable gradient checkpointing")

    ap.add_argument("--max_steps", type=int, default=-1, help="Maximum steps; -1 runs all epochs")
    return ap.parse_args()

# ---------------- dist utils ----------------
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def is_main(rank): return rank == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def all_reduce_mean(t: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(dist.get_world_size())
    return t

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------- utils ----------------
def encode_pairs_split_trunc(tokenizer, queries, docs, *, q_max=256, d_max=256, device="cpu"):
    q_tok = tokenizer(queries, add_special_tokens=False,
                      truncation=True, max_length=q_max, return_tensors=None)
    d_tok = tokenizer(docs, add_special_tokens=False,
                      truncation=True, max_length=d_max, return_tensors=None)
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
    return {"input_ids": input_ids_pad.to(device),
            "attention_mask": attn_mask.to(device)}

def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    if h: return f"{h}h{m}m{s}s"
    if m: return f"{m}m{s}s"
    return f"{s}s"

def batch_stats(Y, lens, max_pairs):
    off = 0; pos_ct=[]; neg_ct=[]; pairs_est=[]
    for L in lens:
        yy = Y[off:off+L]
        p = int((yy==1).sum().item()); n = int((yy==0).sum().item())
        pos_ct.append(p); neg_ct.append(n)
        total_pairs = p*n; pairs_est.append(min(total_pairs, max_pairs) if total_pairs>0 else 0)
        off += L
    return (np.mean(pos_ct) if pos_ct else 0.0,
            np.mean(neg_ct) if neg_ct else 0.0,
            np.mean(pairs_est) if pairs_est else 0.0)

def seq_len_stats(attn_mask: torch.Tensor):
    eff = attn_mask.sum(dim=1).float()
    return float(eff.mean().item()), int(eff.max().item()), int(attn_mask.size(1))

# ---------------- data ----------------
class GroupDataset(Dataset):
    def __init__(self, tsv):
        df = pd.read_csv(tsv, sep="\t",
            usecols=["query","cand_text","group_id","cand_label","group_label"],
            dtype={"cand_label": int, "group_label": float}
        ).fillna({"group_label": -1})
        self.groups = []
        for gid, g in df.groupby("group_id", sort=False):
            q = g["query"].iloc[0]
            docs = g["cand_text"].tolist()
            y = torch.tensor(g["cand_label"].tolist(), dtype=torch.long)
            if (g["group_label"] >= 0).any():
                uniq = g["group_label"].dropna().unique()
                gl = float(uniq[0]) if len(uniq)==1 else (1.0 if (y==1).any().item() else 0.0)
            else:
                gl = 1.0 if (y==1).any().item() else 0.0
            self.groups.append((q, docs, y, gl))
    def __len__(self): return len(self.groups)
    def __getitem__(self, i): return self.groups[i]

def collate_groups(batch):
    Q, D, Y, lens, group_labels = [], [], [], [], []
    for q, docs, y, gl in batch:
        k = len(docs); Q += [q]*k; D += docs; Y.append(y); group_labels.append(gl); lens.append(k)
    Y = torch.cat(Y, dim=0)
    group_labels = torch.tensor(group_labels, dtype=torch.float32)
    return Q, D, Y, lens, group_labels

# ---------------- loss ----------------
def groupwise_loss(scores, cand_labels, lens, group_labels, *,
                   rank_loss="logistic", margin=1.0,
                   alpha=1.0, gamma=0.05,
                   beta_ent=0.2, T=1.0, max_pairs=1024,
                   beta_IRDA=0.1,
                   device="cpu"):
    losses = []
    off = 0
    for L, gl in zip(lens, group_labels):
        s = scores[off:off+L]
        y = cand_labels[off:off+L]
        pos = s[y == 1]
        neg = s[y == 0]
        has_pos = (pos.numel() > 0)
        has_neg = (neg.numel() > 0)

        if has_pos:

            if has_neg:

                group_std = s.std(unbiased=False).clamp_min(1e-8)
                delta = pos[:, None] - neg[None, :]  # (P, N)
                d = delta.reshape(-1)                # (P*N,)

                if d.numel() > max_pairs:
                    idx = torch.randperm(d.numel(), device=device)[:max_pairs]
                    d = d[idx]

                if rank_loss == "logistic":
                    margin_dynamic = (
                        margin +
                        alpha * torch.sigmoid(-d) +
                        gamma / group_std
                    )
                    loss_pairwise = torch.log1p(torch.exp(- (d - margin_dynamic))).mean()
                else:
                    loss_pairwise = torch.clamp(margin - d, min=0).mean()

                # IRDA (only if beta_IRDA > 0)
                if beta_IRDA > 0.0:
                    P_ideal = torch.zeros_like(y, dtype=torch.float32)
                    num_pos = (y == 1).sum().item()
                    if num_pos > 0:
                        P_ideal[y == 1] = 1.0 / num_pos
                    P_model = torch.softmax(s, dim=0)
                    loss_IRDA = torch.sum(P_ideal * torch.log((P_ideal + 1e-12) / (P_model + 1e-12)))
                    total_loss = loss_pairwise + beta_IRDA * loss_IRDA
                else:
                    total_loss = loss_pairwise

                losses.append(total_loss)
            else:

                losses.append(torch.zeros((), device=device))

        else:

            if has_neg:
                if beta_ent > 0.0:

                    p = torch.softmax(neg / T, dim=0)
                    l_ent = (p * torch.log(p * neg.numel() + 1e-12)).sum()
                    losses.append(beta_ent * l_ent)
                else:



                    if rank_loss == "hinge":
                        loss_neg = torch.clamp(neg + margin, min=0).mean()
                    else:  # logistic-like: log(1 + exp(s_i))
                        loss_neg = torch.log1p(torch.exp(neg)).mean()
                    losses.append(loss_neg)
            else:

                losses.append(torch.zeros((), device=device))
        off += L

    return torch.stack(losses).mean()

# ---------------- metrics (DDP reduce) ----------------
@torch.no_grad()
def evaluate(model, tok, loader, device, *, cutoff=10, T=1.0, q_max=256, d_max=256, rank=0,
             clamp=20.0, eps=1e-8):

    was_training = model.training
    model.eval()
    RR_sum = torch.tensor(0.0, device=device); P5_sum = torch.tensor(0.0, device=device)
    P10_sum = torch.tensor(0.0, device=device); pos_groups = torch.tensor(0.0, device=device)
    ent_sum = torch.tensor(0.0, device=device); maxs_sum = torch.tensor(0.0, device=device)
    neg_groups = torch.tensor(0.0, device=device)


    Top1Acc_sum = torch.tensor(0.0, device=device)
    MinPosRank_sum = torch.tensor(0.0, device=device)
    MaxNegRank_sum = torch.tensor(0.0, device=device)
    RankGap_sum = torch.tensor(0.0, device=device)
    KLDiv_sum = torch.tensor(0.0, device=device)
    kl_groups = torch.tensor(0.0, device=device)

    for Q, D, Y, lens, CDs in loader:
        enc = encode_pairs_split_trunc(tok, Q, D, q_max=q_max, d_max=d_max, device=device)
        s_gpu = model(enc["input_ids"], enc["attention_mask"].long(), mode="rerank")["scores"].squeeze(-1)


        s = torch.nan_to_num(s_gpu, nan=0.0, posinf=1e6, neginf=-1e6).detach().cpu()
        Y_cpu = Y.cpu()


        off = 0
        norm_list = []
        for L in lens:
            sg = s[off:off+L].float()
            if L > 1:
                mu = sg.mean()
                var = (sg - mu).pow(2).mean()
                if var < eps:
                    sg = torch.zeros_like(sg)
                else:
                    std = var.sqrt()
                    sg = (sg - mu) / (std + eps)
            sg = sg.clamp_(-clamp, clamp)
            norm_list.append(sg)
            off += L
        s = torch.cat(norm_list, dim=0) if len(norm_list) > 1 else norm_list[0]


        off = 0
        for L, cd in zip(lens, CDs):
            ss = s[off:off+L]; yy = Y_cpu[off:off+L]
            k = min(cutoff, L)

            if (yy==1).any():
                order = torch.argsort(ss, descending=True)
                top = yy[order[:k]]


                idx = (top==1).nonzero(as_tuple=True)[0]
                rr = 1.0/float(idx[0].item()+1) if idx.numel()>0 else 0.0
                RR_sum += rr
                P5_sum  += float(top[:min(5,k)].float().mean())
                P10_sum += float(top[:min(10,k)].float().mean())
                pos_groups += 1.0


                Top1Acc_sum += float(yy[order[0]] == 1)
                min_pos_rank = order[yy == 1].min() + 1
                MinPosRank_sum += float(min_pos_rank)
                if (yy == 0).any():
                    max_neg_rank = order[yy == 0].max() + 1
                    MaxNegRank_sum += float(max_neg_rank)
                    RankGap_sum += float(max_neg_rank - min_pos_rank)


                ss_raw = s_gpu[off:off+L].float().detach().cpu()
                P_ideal = torch.zeros_like(yy, dtype=torch.float32)
                num_pos = (yy == 1).sum().item()
                if num_pos > 0:
                    P_ideal[yy == 1] = 1.0 / num_pos
                P_model = torch.softmax(ss_raw, dim=0)
                kl_div = torch.sum(P_ideal * torch.log((P_ideal + 1e-12) / (P_model + 1e-12)))
                KLDiv_sum += float(kl_div)
                kl_groups += 1.0
            else:

                p = torch.softmax(ss / max(T, eps), dim=0)
                ent_sum  += float((p * torch.log(p * L + eps)).sum())
                maxs_sum += float(ss.max().item())
                neg_groups += 1.0
            off += L

    if dist.is_available() and dist.is_initialized():
        for t in [RR_sum, P5_sum, P10_sum, pos_groups, ent_sum, maxs_sum, neg_groups,
                  Top1Acc_sum, MinPosRank_sum, MaxNegRank_sum, RankGap_sum, KLDiv_sum, kl_groups]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    metrics = {}
    if pos_groups.item() > 0:
        metrics.update({
            "MRR@10": (RR_sum/pos_groups).item(),
            "P@5": (P5_sum/pos_groups).item(),
            "P@10": (P10_sum/pos_groups).item(),
            "PosGroups": int(pos_groups.item()),
            "Top1_Acc": (Top1Acc_sum/pos_groups).item(),
            "Min_Pos_Rank": (MinPosRank_sum/pos_groups).item(),
            "Max_Neg_Rank": (MaxNegRank_sum/pos_groups).item() if pos_groups.item() > 0 else 0.0,
            "Rank_Gap": (RankGap_sum/pos_groups).item() if pos_groups.item() > 0 else 0.0,
            "KL_Div": (KLDiv_sum/kl_groups).item() if kl_groups.item() > 0 else 0.0,
        })
    if neg_groups.item() > 0:
        metrics.update({
            "Neg_KL(mean)": (ent_sum/neg_groups).item(),
            "Neg_MaxScore(mean)": (maxs_sum/neg_groups).item(),
            "NegGroups": int(neg_groups.item())
        })


    if was_training:
        model.train()

    return metrics

@torch.no_grad()
def _log_raw_scores(Q, D, Y, lens, scores, step, epoch):
    """
    Summarize scores for mixed and all-negative groups.
    """
    scores = scores.detach().cpu()
    Y_cpu  = Y.cpu()
    off = 0
    pos_groups, neg_groups = 0, 0
    pos_scores, neg_scores = [], []
    pure_neg_scores = []

    for L in lens:
        grp_s = scores[off:off+L]
        grp_y = Y_cpu[off:off+L]
        if (grp_y == 1).any():
            pos_groups += 1
            pos_scores.extend(grp_s[grp_y == 1].tolist())
            neg_scores.extend(grp_s[grp_y == 0].tolist())
        else:
            neg_groups += 1
            pure_neg_scores.extend(grp_s.tolist())
        off += L


    print(f"[Step {step}]  "
          f"PosGroups={pos_groups}  "
          f"PosScores={pos_scores}  "
          f"NegScores={neg_scores}  |  "
          f"NegGroups={neg_groups}  "
          f"PureNegScores={pure_neg_scores}")

# ---------------- main ----------------
def main():
    args = get_args()
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main(rank): os.makedirs(args.out_dir, exist_ok=True)
    barrier()


    logger = TrainLogger(args.out_dir, enable=is_main(rank))

    if is_main(rank):
        print("="*60)
        print("Training Configuration (DDP):")
        print(f"  World size: {world_size} | Rank: {rank} | Local rank: {local_rank}")
        print(f"  Model: {args.model_path}")
        print(f"  Data: {args.tsv}")
        print(f"  Output: {args.out_dir}")
        print(f"  Epochs: {args.epochs} | Batch groups/GPU: {args.batch_groups}")
        print(f"  LR: {args.lr} | Weight decay: {args.weight_decay}")
        print(f"  Rank loss: {args.rank_loss} | Margin: {args.margin}")
        print(f"  Beta ent: {args.beta_ent} | Temp: {args.temp}")
        print(f"  Max pairs: {args.max_pairs} | Grad clip: {args.grad_clip}")
        print(f"  Warmup ratio: {args.warmup_ratio} | Seed: {args.seed}")
        print(f"  q_max: {args.q_max} | d_max: {args.d_max}")
        print(f"  Alpha: {args.alpha} | Gamma: {args.gamma} | Beta_IRDA: {args.beta_IRDA}")


        if args.use_lora:
            print(f"  LoRA: Enabled | r: {args.lora_r} | alpha: {args.lora_alpha} | dropout: {args.lora_dropout}")
            print(f"  Target modules: {args.target_modules}")
        else:
            print(f"  LoRA: Disabled")
        print("="*60)


    if is_main(rank):
        print(f"[INFO] Setting model initialization seed to {args.seed} for all ranks")


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)


    model = Qwen3ForEmbedding.from_pretrained(args.model_path)


    if args.grad_checkpoint:
        print("[INFO] Enabling Gradient Checkpointing to save memory...")



        try:
            model.gradient_checkpointing_enable()
        except ValueError:

            if hasattr(model, "model"):
                model.model.gradient_checkpointing_enable()

                if hasattr(model, "config"):
                    model.config.gradient_checkpointing = True
            else:

                print("[WARN] Forcing supports_gradient_checkpointing=True")
                model.supports_gradient_checkpointing = True
                model.gradient_checkpointing_enable()


        if hasattr(model, "config"):
            model.config.use_cache = False


        if args.use_lora:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # ==========================================================



    if hasattr(model, "score_head"):
        torch.manual_seed(args.seed)
        with torch.no_grad():
            model.score_head.weight.zero_()

            nn.init.normal_(model.score_head.weight, mean=0.0, std=0.01)
            if model.score_head.bias is not None:
                nn.init.zeros_(model.score_head.bias)


        model.score_head.weight.requires_grad = True
        if model.score_head.bias is not None:
            model.score_head.bias.requires_grad = True


    if args.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=["score_head"]
        )

        model = get_peft_model(model, lora_config)

        if is_main(rank):
            model.print_trainable_parameters()


    try:
        model.model.config.attn_implementation = "sdpa"
    except Exception:
        pass


    if is_main(rank):

        head = None
        if hasattr(model, "score_head"):
            head = model.score_head
        elif hasattr(model, "module") and hasattr(model.module, "score_head"):
            head = model.module.score_head
        elif hasattr(model, "base_model") and hasattr(model.base_model, "score_head"):
            head = model.base_model.score_head

        if head is not None:
            print(f"[Check] score_head initialized. Weight std: {head.weight.data.std().item():.6f}")

    set_seed(args.seed + rank)

    model.to(device)

    if is_main(rank):
        print("\n[DEBUG] Model parameters (focus on score head & LoRA):")
        for name, param in model.named_parameters():
            if "score" in name or "head" in name or "classifier" in name or "lm_head" in name or "lora_" in name.lower():
                print(f"  {name}: requires_grad={param.requires_grad}, mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        print()


    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )


    model.train()

    ds = GroupDataset(args.tsv)
    n = len(ds); split = int(0.9*n)
    if is_main(rank): print(f"Dataset: {n} groups (train: {split}, valid: {n-split})")

    train_subset = torch.utils.data.Subset(ds, list(range(split)))
    valid_subset = torch.utils.data.Subset(ds, list(range(split, n)))

    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    valid_sampler = DistributedSampler(valid_subset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    tr_ld = DataLoader(train_subset, batch_size=args.batch_groups, sampler=train_sampler, collate_fn=collate_groups)
    va_ld = DataLoader(valid_subset, batch_size=args.batch_groups, sampler=valid_sampler, collate_fn=collate_groups)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(tr_ld)
    computed_total_steps = args.epochs * steps_per_epoch

    if args.max_steps > 0:
        total_steps = args.max_steps


        if total_steps < computed_total_steps:
             if is_main(rank):
                 print(f"[Info] Overriding total_steps from {computed_total_steps} to {total_steps} due to --max_steps")
    else:
        total_steps = computed_total_steps

    warm_steps = int(total_steps * args.warmup_ratio)
    sched = get_linear_schedule_with_warmup(optim, warm_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    best_ndcg = -1.0
    best_dir = os.path.join(args.out_dir, "best"); last_dir = os.path.join(args.out_dir, "last")
    if is_main(rank):
        os.makedirs(best_dir, exist_ok=True); os.makedirs(last_dir, exist_ok=True)

    steps_done = 0; ema_step_sec = None; step = 0; run_loss = 0.0
    start_time = time.time()

    stop_training = False


    SCORE_LOG_EVERY = max(1, args.log_steps // 2)

    for ep in range(1, args.epochs+1):
        if stop_training: break


        model.train()
        train_sampler.set_epoch(ep)
        ep_start = time.time()

        for Q, D, Y, lens, group_labels in tr_ld:
            if args.max_steps > 0 and step >= args.max_steps:
                stop_training = True
                break
            t0 = time.time()

            enc = encode_pairs_split_trunc(tok, Q, D, q_max=args.q_max, d_max=args.d_max, device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(enc["input_ids"], enc["attention_mask"].long(), mode="rerank")
                scores = torch.nan_to_num(out["scores"].squeeze(-1), nan=-1e9)
                loss = groupwise_loss(scores, Y.to(device), lens, group_labels.to(device),
                      rank_loss=args.rank_loss, margin=args.margin,
                      alpha=args.alpha, gamma=args.gamma,
                      beta_ent=args.beta_ent, T=args.temp,
                      max_pairs=args.max_pairs, device=device,
                      beta_IRDA=args.beta_IRDA)
            scaler.scale(loss).backward()


            scaler.unscale_(optim)

            grad_norm = torch.tensor(0.0, device=device)
            if is_main(rank):

                grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                if grads:
                    grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)


            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optim); scaler.update(); sched.step()
            optim.zero_grad(set_to_none=True)

            loss_detached = float(loss.detach())
            run_loss += loss_detached; step += 1; steps_done += 1


            if is_main(rank) and step % SCORE_LOG_EVERY == 0:
                _log_raw_scores(Q, D, Y, lens, scores, step, ep)

            step_sec = time.time() - t0
            if ema_step_sec is None: ema_step_sec = step_sec
            else: ema_step_sec = 0.9*ema_step_sec + 0.1*step_sec
            eta_sec = max(0.0, (total_steps-steps_done) * ema_step_sec)

            if is_main(rank) and step % args.log_steps == 0:
                avg_len, max_eff_len, pad_len = seq_len_stats(enc["attention_mask"])
                avg_pos, avg_neg, avg_pairs = batch_stats(Y, lens, args.max_pairs)
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device) / (1024**3)
                    mem_info = f"GPU{local_rank} mem alloc/resv: {alloc:.1f}G/{reserved:.1f}G"
                else:
                    alloc = reserved = 0.0
                    mem_info = "GPU mem: N/A"
                current_lr = sched.get_last_lr()[0]

                grad_norm_cpu = float(grad_norm.item()) if is_main(rank) else 0.0

                print(
                    f"[Main] ep {ep} step {step}/{total_steps} "
                    f"loss={run_loss/args.log_steps:.4f} lr={current_lr:.2e} "
                    f"grad_norm={grad_norm_cpu:.4f} "
                    f"| seq_len avg/max/pad={avg_len:.1f}/{max_eff_len}/{pad_len} "
                    f"| grp avg pos/neg={avg_pos:.1f}/{avg_neg:.1f} pairs~{avg_pairs:.0f} "
                    f"| step={step_sec:.2f}s ema={ema_step_sec:.2f}s ETA={fmt_time(eta_sec)} "
                    f"| {mem_info}"
                )


                logger.log_step(
                    epoch=ep,
                    global_step=step,
                    loss_avg=run_loss/args.log_steps,
                    lr=current_lr,
                    grad_norm=grad_norm_cpu,
                    seq_len_avg=avg_len,
                    seq_len_max=max_eff_len,
                    pad_len=pad_len,
                    avg_pos=avg_pos,
                    avg_neg=avg_neg,
                    avg_pairs=avg_pairs,
                    step_sec=step_sec,
                    ema_step_sec=ema_step_sec,
                    eta_sec=eta_sec,
                    gpu_mem_alloc_gb=alloc,
                    gpu_mem_reserved_gb=reserved,
                )
                logger.plot_curves()
                run_loss = 0.0

            # del enc, out, scores
            # if torch.cuda.is_available(): torch.cuda.empty_cache()
            if args.max_steps > 0 and step >= args.max_steps:
                if is_main(rank):
                    print(f"[Stop] Reached max_steps={args.max_steps}. Exiting training loop.")
                stop_training = True
                break

        ep_time = time.time() - ep_start
        barrier()
        valid_sampler.set_epoch(ep)
        metrics = evaluate(model, tok, va_ld, device,
                           cutoff=10, T=args.temp, q_max=args.q_max, d_max=args.d_max, rank=rank)

        if is_main(rank):
            print(f"\n==> Epoch {ep} completed in {fmt_time(ep_time)}")
            print("Validation Metrics:")
            if "MRR@10" in metrics:
                print(f"  MRR@10: {metrics['MRR@10']:.4f} | P@5: {metrics['P@5']:.4f} | P@10: {metrics['P@10']:.4f} | PosGroups: {metrics['PosGroups']}")
                print(f"  Top1_Acc: {metrics['Top1_Acc']:.4f} | Min_Pos_Rank: {metrics['Min_Pos_Rank']:.2f} | Max_Neg_Rank: {metrics['Max_Neg_Rank']:.2f} | Rank_Gap: {metrics['Rank_Gap']:.2f} | KL_Div: {metrics['KL_Div']:.4f}")
            if "NegGroups" in metrics:
                print(f"  Neg_KL(mean): {metrics['Neg_KL(mean)']:.4f} | Neg_MaxScore(mean): {metrics['Neg_MaxScore(mean)']:.4f} | NegGroups: {metrics['NegGroups']}")
            print("-"*60)


            logger.log_epoch(ep, ep_time, metrics)
            logger.plot_curves()

            score_for_best = metrics.get("MRR@10", 0.0)

            if score_for_best > best_ndcg:
                best_ndcg = score_for_best
                if is_main(rank):
                    if args.use_lora:

                        model.module.save_pretrained(best_dir)
                    else:
                        model.module.save_pretrained(best_dir)
                    tok.save_pretrained(best_dir)
                    torch.save({"epoch": ep, "best_MRR@10": best_ndcg}, os.path.join(best_dir, "trainer_state.pt"))
                    print(f"[save] best -> {best_dir} (MRR@10: {best_ndcg:.4f})")


            if is_main(rank):
                if args.use_lora:

                    model.module.save_pretrained(last_dir)
                else:
                    model.module.save_pretrained(last_dir)
                tok.save_pretrained(last_dir)
                torch.save({"epoch": ep, "MRR@10": score_for_best}, os.path.join(last_dir, "trainer_state.pt"))

if __name__ == "__main__":
    main()
