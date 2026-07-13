# train_reranker_group_label_ddp.py
# pip install torch transformers pandas matplotlib peft
import os, math, argparse, random, time, csv
from pathlib import Path
import sys
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch.distributed as dist
import io, json


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_DIR))
from modeling_qwen3_embed import Qwen3ForEmbedding  # noqa: E402

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
                        P_×Ýz¶‰žËkºwµçU¹Ñ}ÍÕ´½¹•}É½ÕÁÌ¤¹¥Ñ•´ ¤°(€€€€€€€€€€€€‰9•}5…áM½É”¡µ•…¸¤ˆè€¡µ…áÍ}ÍÕ´½¹•}É½ÕÁÌ¤¹¥Ñ•´ ¤°(€€€€€€€€€€€€‰9•É½ÕÁÌˆè¥¹Ð¡¹•}É½ÕÁÌ¹¥Ñ•´ ¤¤(€€€€€€€ô¤(((€€€¥˜Ý…Í}ÑÉ…¥¹¥¹œè(€€€€€€€µ½‘•°¹ÑÉ…¥¸ ¤((€€€É•ÑÕÉ¸µ•ÑÉ¥Ì()Ñ½É ¹¹½}É… ¤)‘•˜}±½}É…Ý}Í½É•Ì¡D°°d°±•¹Ì°Í½É•Ì°ÍÑ•À°•Á½ ¤è(€€€€ˆˆˆ(€€€MÕµµ…É¥é”Í½É•Ì™½Èµ¥á•…¹…±°µ¹•…Ñ¥Ù”É½ÕÁÌ¸(€€€€ˆˆˆ(€€€Í½É•Ì€ôÍ½É•Ì¹‘•Ñ…  ¤¹ÁÔ ¤(€€€e}ÁÔ€€ôd¹ÁÔ ¤(€€€½™˜€ô€À(€€€Á½Í}É½ÕÁÌ°¹•}É½ÕÁÌ€ô€À°€À(€€€Á½Í}Í½É•Ì°¹•}Í½É•Ì€ômt°mt(€€€ÁÕÉ•}¹•}Í½É•Ì€ômt((€€€™½È0¥¸±•¹Ìè(€€€€€€€ÉÁ}Ì€ôÍ½É•Ím½™˜é½™˜­1t(€€€€€€€ÉÁ}ä€ôe}ÁÕm½™˜é½™˜­1t(€€€€€€€¥˜€¡ÉÁ}ä€ôô€Ä¤¹…¹ä ¤è(€€€€€€€€€€€Á½Í}É½ÕÁÌ€¬ô€Ä(€€€€€€€€€€€Á½Í}Í½É•Ì¹•áÑ•¹¡ÉÁ}ÍmÉÁ}ä€ôô€Åt¹Ñ½±¥ÍÐ ¤¤(€€€€€€€€€€€¹•}Í½É•Ì¹•áÑ•¹¡ÉÁ}ÍmÉÁ}ä€ôô€Át¹Ñ½±¥ÍÐ ¤¤(€€€€€€€•±Í”è(€€€€€€€€€€€¹•}É½ÕÁÌ€¬ô€Ä(€€€€€€€€€€€ÁÕÉ•}¹•}Í½É•Ì¹•áÑ•¹¡ÉÁ}Ì¹Ñ½±¥ÍÐ ¤¤(€€€€€€€½™˜€¬ô0(((€€€ÁÉ¥¹Ð¡˜‰mMÑ•ÀíÍÑ•Áõt€€ˆ(€€€€€€€€€˜‰A½ÍÉ½ÕÁÌõíÁ½Í}É½ÕÁÍô€€ˆ(€€€€€€€€€˜‰A½ÍM½É•ÌõíÁ½Í}Í½É•Íô€€ˆ(€€€€€€€€€˜‰9•M½É•Ìõí¹•}Í½É•Íô€ð€€ˆ(€€€€€€€€€˜‰9•É½ÕÁÌõí¹•}É½ÕÁÍô€€ˆ(€€€€€€€€€˜‰AÕÉ•9•M½É•ÌõíÁÕÉ•}¹•}Í½É•Íôˆ¤((Œ€´´´´´´´´´´´´´´´´µ…¥¸€´´´´´´´´´´´´´´´´)‘•˜µ…¥¸ ¤è(€€€…ÉÌ€ô•Ñ}…ÉÌ ¤(€€€É…¹¬°Ý½É±‘}Í¥é”°±½…±}É…¹¬€ôÍ•ÑÕÁ}‘‘À ¤(€€€‘•Ù¥”€ôÑ½É ¹‘•Ù¥”¡˜‰Õ‘„éí±½…±}É…¹­ôˆ¥˜Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤•±Í”€‰ÁÔˆ¤(€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è½Ì¹µ…­•‘¥ÉÌ¡…ÉÌ¹½ÕÑ}‘¥È°•á¥ÍÑ}½¬õQÉÕ”¤(€€€‰…ÉÉ¥•È ¤(((€€€±½•È€ôQÉ…¥¹1½•È¡…ÉÌ¹½ÕÑ}‘¥È°•¹…‰±”õ¥Í}µ…¥¸¡É…¹¬¤¤((€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€ÁÉ¥¹Ð ˆôˆ¨ØÀ¤(€€€€€€€ÁÉ¥¹Ð ‰QÉ…¥¹¥¹œ½¹™¥ÕÉ…Ñ¥½¸€¡@¤èˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€]½É±Í¥é”èíÝ½É±‘}Í¥é•ôðI…¹¬èíÉ…¹­ôð1½…°É…¹¬èí±½…±}É…¹­ôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€5½‘•°èí…ÉÌ¹µ½‘•±}Á…Ñ¡ôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€…Ñ„èí…ÉÌ¹ÑÍÙôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€=ÕÑÁÕÐèí…ÉÌ¹½ÕÑ}‘¥Éôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€Á½¡Ìèí…ÉÌ¹•Á½¡Íôð	…Ñ É½ÕÁÌ½ATèí…ÉÌ¹‰…Ñ¡}É½ÕÁÍôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€1Hèí…ÉÌ¹±Éôð]•¥¡Ð‘•…äèí…ÉÌ¹Ý•¥¡Ñ}‘•…åôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€I…¹¬±½ÍÌèí…ÉÌ¹É…¹­}±½ÍÍôð5…É¥¸èí…ÉÌ¹µ…É¥¹ôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€	•Ñ„•¹Ðèí…ÉÌ¹‰•Ñ…}•¹ÑôðQ•µÀèí…ÉÌ¹Ñ•µÁôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€5…àÁ…¥ÉÌèí…ÉÌ¹µ…á}Á…¥ÉÍôðÉ…±¥Àèí…ÉÌ¹É…‘}±¥Áôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€]…ÉµÕÀÉ…Ñ¥¼èí…ÉÌ¹Ý…ÉµÕÁ}É…Ñ¥½ôðM••èí…ÉÌ¹Í••‘ôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€Å}µ…àèí…ÉÌ¹Å}µ…áôð‘}µ…àèí…ÉÌ¹‘}µ…áôˆ¤(€€€€€€€ÁÉ¥¹Ð¡˜ˆ€±Á¡„èí…ÉÌ¹…±Á¡…ôð…µµ„èí…ÉÌ¹…µµ…ôð	•Ñ…}%Ièí…ÉÌ¹‰•Ñ…}%Iôˆ¤(((€€€€€€€¥˜…ÉÌ¹ÕÍ•}±½É„è(€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€1½Iè¹…‰±•ðÈèí…ÉÌ¹±½É…}Éôð…±Á¡„èí…ÉÌ¹±½É…}…±Á¡…ôð‘É½Á½ÕÐèí…ÉÌ¹±½É…}‘É½Á½ÕÑôˆ¤(€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€Q…É•Ðµ½‘Õ±•Ìèí…ÉÌ¹Ñ…É•Ñ}µ½‘Õ±•Íôˆ¤(€€€€€€€•±Í”è(€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€1½Iè¥Í…‰±•ˆ¤(€€€€€€€ÁÉ¥¹Ð ˆôˆ¨ØÀ¤(((€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€ÁÉ¥¹Ð¡˜‰m%9=tM•ÑÑ¥¹œµ½‘•°¥¹¥Ñ¥…±¥é…Ñ¥½¸Í••Ñ¼í…ÉÌ¹Í••‘ô™½È…±°É…¹­Ìˆ¤(((€€€Ñ½É ¹µ…¹Õ…±}Í••¡…ÉÌ¹Í••¤(€€€Ñ½É ¹Õ‘„¹µ…¹Õ…±}Í••‘}…±°¡…ÉÌ¹Í••¤(€€€¹À¹É…¹‘½´¹Í••¡…ÉÌ¹Í••¤(€€€É…¹‘½´¹Í••¡…ÉÌ¹Í••¤(((€€€Ñ½¬€ôÕÑ½Q½­•¹¥é•È¹™É½µ}ÁÉ•ÑÉ…¥¹•¡…ÉÌ¹µ½‘•±}Á…Ñ ¤(((€€€µ½‘•°€ôEÝ•¸Í½Éµ‰•‘‘¥¹œ¹™É½µ}ÁÉ•ÑÉ…¥¹•¡…ÉÌ¹µ½‘•±}Á…Ñ ¤(((€€€¥˜…ÉÌ¹É…‘}¡•­Á½¥¹Ðè(€€€€€€€ÁÉ¥¹Ð ‰m%9=t¹…‰±¥¹œÉ…‘¥•¹Ð¡•­Á½¥¹Ñ¥¹œÑ¼Í…Ù”µ•µ½Éä¸¸¸ˆ¤((((€€€€€€€ÑÉäè(€€€€€€€€€€€µ½‘•°¹É…‘¥•¹Ñ}¡•­Á½¥¹Ñ¥¹}•¹…‰±” ¤(€€€€€€€•á•ÁÐY…±Õ•ÉÉ½Èè((€€€€€€€€€€€¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰µ½‘•°ˆ¤è(€€€€€€€€€€€€€€€µ½‘•°¹µ½‘•°¹É…‘¥•¹Ñ}¡•­Á½¥¹Ñ¥¹}•¹…‰±” ¤((€€€€€€€€€€€€€€€¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰½¹™¥œˆ¤è(€€€€€€€€€€€€€€€€€€€µ½‘•°¹½¹™¥œ¹É…‘¥•¹Ñ}¡•­Á½¥¹Ñ¥¹œ€ôQÉÕ”(€€€€€€€€€€€•±Í”è((€€€€€€€€€€€€€€€ÁÉ¥¹Ð ‰m]I9t½É¥¹œÍÕÁÁ½ÉÑÍ}É…‘¥•¹Ñ}¡•­Á½¥¹Ñ¥¹œõQÉÕ”ˆ¤(€€€€€€€€€€€€€€€µ½‘•°¹ÍÕÁÁ½ÉÑÍ}É…‘¥•¹Ñ}¡•­Á½¥¹Ñ¥¹œ€ôQÉÕ”(€€€€€€€€€€€€€€€µ½‘•°¹É…‘¥•¹Ñ}¡•­Á½¥¹Ñ¥¹}•¹…‰±” ¤(((€€€€€€€¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰½¹™¥œˆ¤è(€€€€€€€€€€€µ½‘•°¹½¹™¥œ¹ÕÍ•}…¡”€ô…±Í”(((€€€€€€€¥˜…ÉÌ¹ÕÍ•}±½É„è(€€€€€€€€€€€¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰•¹…‰±•}¥¹ÁÕÑ}É•ÅÕ¥É•}É…‘Ìˆ¤è(€€€€€€€€€€€€€€€µ½‘•°¹•¹…‰±•}¥¹ÁÕÑ}É•ÅÕ¥É•}É…‘Ì ¤(€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€‘•˜µ…­•}¥¹ÁÕÑÍ}É•ÅÕ¥É•}É…¡µ½‘Õ±”°¥¹ÁÕÐ°½ÕÑÁÕÐ¤è(€€€€€€€€€€€€€€€€€€€½ÕÑÁÕÐ¹É•ÅÕ¥É•Í}É…‘|¡QÉÕ”¤(€€€€€€€€€€€€€€€µ½‘•°¹•Ñ}¥¹ÁÕÑ}•µ‰•‘‘¥¹Ì ¤¹É•¥ÍÑ•É}™½ÉÝ…É‘}¡½½¬¡µ…­•}¥¹ÁÕÑÍ}É•ÅÕ¥É•}É…¤(€€€€Œ€ôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôôô((((€€€¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰Í½É•}¡•…ˆ¤è(€€€€€€€Ñ½É ¹µ…¹Õ…±}Í••¡…ÉÌ¹Í••¤(€€€€€€€Ý¥Ñ Ñ½É ¹¹½}É… ¤è(€€€€€€€€€€€µ½‘•°¹Í½É•}¡•…¹Ý•¥¡Ð¹é•É½| ¤((€€€€€€€€€€€¹¸¹¥¹¥Ð¹¹½Éµ…±|¡µ½‘•°¹Í½É•}¡•…¹Ý•¥¡Ð°µ•…¸ôÀ¸À°ÍÑôÀ¸ÀÄ¤(€€€€€€€€€€€¥˜µ½‘•°¹Í½É•}¡•…¹‰¥…Ì¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€¹¸¹¥¹¥Ð¹é•É½Í|¡µ½‘•°¹Í½É•}¡•…¹‰¥…Ì¤(((€€€€€€€µ½‘•°¹Í½É•}¡•…¹Ý•¥¡Ð¹É•ÅÕ¥É•Í}É…€ôQÉÕ”(€€€€€€€¥˜µ½‘•°¹Í½É•}¡•…¹‰¥…Ì¥Ì¹½Ð9½¹”è(€€€€€€€€€€€µ½‘•°¹Í½É•}¡•…¹‰¥…Ì¹É•ÅÕ¥É•Í}É…€ôQÉÕ”(((€€€¥˜…ÉÌ¹ÕÍ•}±½É„è(€€€€€€€±½É…}½¹™¥œ€ô1½É…½¹™¥œ (€€€€€€€€€€€Ñ…Í­}ÑåÁ”õQ…Í­QåÁ”¹QUI}aQIQ%=8°(€€€€€€€€€€€Èõ…ÉÌ¹±½É…}È°(€€€€€€€€€€€±½É…}…±Á¡„õ…ÉÌ¹±½É…}…±Á¡„°(€€€€€€€€€€€Ñ…É•Ñ}µ½‘Õ±•Ìõ…ÉÌ¹Ñ…É•Ñ}µ½‘Õ±•Ì°(€€€€€€€€€€€±½É…}‘É½Á½ÕÐõ…ÉÌ¹±½É…}‘É½Á½ÕÐ°(€€€€€€€€€€€‰¥…Ìô‰¹½¹”ˆ°(€€€€€€€€€€€µ½‘Õ±•Í}Ñ½}Í…Ù”õl‰Í½É•}¡•…‰t(€€€€€€€€¤((€€€€€€€µ½‘•°€ô•Ñ}Á•™Ñ}µ½‘•°¡µ½‘•°°±½É…}½¹™¥œ¤((€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€€€€€µ½‘•°¹ÁÉ¥¹Ñ}ÑÉ…¥¹…‰±•}Á…É…µ•Ñ•ÉÌ ¤(((€€€ÑÉäè(€€€€€€€µ½‘•°¹µ½‘•°¹½¹™¥œ¹…ÑÑ¹}¥µÁ±•µ•¹Ñ…Ñ¥½¸€ô€‰Í‘Á„ˆ(€€€•á•ÁÐá•ÁÑ¥½¸è(€€€€€€€Á…ÍÌ(((€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è((€€€€€€€¡•…€ô9½¹”(€€€€€€€¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰Í½É•}¡•…ˆ¤è(€€€€€€€€€€€¡•…€ôµ½‘•°¹Í½É•}¡•…(€€€€€€€•±¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰µ½‘Õ±”ˆ¤…¹¡…Í…ÑÑÈ¡µ½‘•°¹µ½‘Õ±”°€‰Í½É•}¡•…ˆ¤è(€€€€€€€€€€€¡•…€ôµ½‘•°¹µ½‘Õ±”¹Í½É•}¡•…(€€€€€€€•±¥˜¡…Í…ÑÑÈ¡µ½‘•°°€‰‰…Í•}µ½‘•°ˆ¤…¹¡…Í…ÑÑÈ¡µ½‘•°¹‰…Í•}µ½‘•°°€‰Í½É•}¡•…ˆ¤è(€€€€€€€€€€€¡•…€ôµ½‘•°¹‰…Í•}µ½‘•°¹Í½É•}¡•…((€€€€€€€¥˜¡•…¥Ì¹½Ð9½¹”è(€€€€€€€€€€€ÁÉ¥¹Ð¡˜‰m¡•­tÍ½É•}¡•…¥¹¥Ñ¥…±¥é•¸]•¥¡ÐÍÑèí¡•…¹Ý•¥¡Ð¹‘…Ñ„¹ÍÑ ¤¹¥Ñ•´ ¤è¸Ù™ôˆ¤((€€€Í•Ñ}Í••¡…ÉÌ¹Í••€¬É…¹¬¤((€€€µ½‘•°¹Ñ¼¡‘•Ù¥”¤((€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€ÁÉ¥¹Ð ‰q¹m	Ut5½‘•°Á…É…µ•Ñ•ÉÌ€¡™½ÕÌ½¸Í½É”¡•…€˜1½I¤èˆ¤(€€€€€€€™½È¹…µ”°Á…É…´¥¸µ½‘•°¹¹…µ•‘}Á…É…µ•Ñ•ÉÌ ¤è(€€€€€€€€€€€¥˜€‰Í½É”ˆ¥¸¹…µ”½È€‰¡•…ˆ¥¸¹…µ”½È€‰±…ÍÍ¥™¥•Èˆ¥¸¹…µ”½È€‰±µ}¡•…ˆ¥¸¹…µ”½È€‰±½É…|ˆ¥¸¹…µ”¹±½Ý•È ¤è(€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€í¹…µ•ôèÉ•ÅÕ¥É•Í}É…õíÁ…É…´¹É•ÅÕ¥É•Í}É…‘ô°µ•…¸õíÁ…É…´¹‘…Ñ„¹µ•…¸ ¤¹¥Ñ•´ ¤è¸Ù™ô°ÍÑõíÁ…É…´¹‘…Ñ„¹ÍÑ ¤¹¥Ñ•´ ¤è¸Ù™ôˆ¤(€€€€€€€ÁÉ¥¹Ð ¤(((€€€µ½‘•°€ôÑ½É ¹¹¸¹Á…É…±±•°¹¥ÍÑÉ¥‰ÕÑ•‘…Ñ…A…É…±±•° (€€€€€€€µ½‘•°°‘•Ù¥•}¥‘Ìõm±½…±}É…¹­t°½ÕÑÁÕÑ}‘•Ù¥”õ±½…±}É…¹¬°™¥¹‘}Õ¹ÕÍ•‘}Á…É…µ•Ñ•ÉÌõ…±Í”(€€€€¤(((€€€µ½‘•°¹ÑÉ…¥¸ ¤((€€€‘Ì€ôÉ½ÕÁ…Ñ…Í•Ð¡…ÉÌ¹ÑÍØ¤(€€€¸€ô±•¸¡‘Ì¤ìÍÁ±¥Ð€ô¥¹Ð À¸ä©¸¤(€€€¥˜¥Í}µ…¥¸¡É…¹¬¤èÁÉ¥¹Ð¡˜‰…Ñ…Í•Ðèí¹ôÉ½ÕÁÌ€¡ÑÉ…¥¸èíÍÁ±¥Ñô°Ù…±¥èí¸µÍÁ±¥Ñô¤ˆ¤((€€€ÑÉ…¥¹}ÍÕ‰Í•Ð€ôÑ½É ¹ÕÑ¥±Ì¹‘…Ñ„¹MÕ‰Í•Ð¡‘Ì°±¥ÍÐ¡É…¹”¡ÍÁ±¥Ð¤¤¤(€€€Ù…±¥‘}ÍÕ‰Í•Ð€ôÑ½É ¹ÕÑ¥±Ì¹‘…Ñ„¹MÕ‰Í•Ð¡‘Ì°±¥ÍÐ¡É…¹”¡ÍÁ±¥Ð°¸¤¤¤((€€€ÑÉ…¥¹}Í…µÁ±•È€ô¥ÍÑÉ¥‰ÕÑ•‘M…µÁ±•È¡ÑÉ…¥¹}ÍÕ‰Í•Ð°¹Õµ}É•Á±¥…ÌõÝ½É±‘}Í¥é”°É…¹¬õÉ…¹¬°Í¡Õ™™±”õ…±Í”°‘É½Á}±…ÍÐõ…±Í”¤(€€€Ù…±¥‘}Í…µÁ±•È€ô¥ÍÑÉ¥‰ÕÑ•‘M…µÁ±•È¡Ù…±¥‘}ÍÕ‰Í•Ð°¹Õµ}É•Á±¥…ÌõÝ½É±‘}Í¥é”°É…¹¬õÉ…¹¬°Í¡Õ™™±”õ…±Í”°‘É½Á}±…ÍÐõ…±Í”¤((€€€ÑÉ}±€ô…Ñ…1½…‘•È¡ÑÉ…¥¹}ÍÕ‰Í•Ð°‰…Ñ¡}Í¥é”õ…ÉÌ¹‰…Ñ¡}É½ÕÁÌ°Í…µÁ±•ÈõÑÉ…¥¹}Í…µÁ±•È°½±±…Ñ•}™¸õ½±±…Ñ•}É½ÕÁÌ¤(€€€Ù…}±€ô…Ñ…1½…‘•È¡Ù…±¥‘}ÍÕ‰Í•Ð°‰…Ñ¡}Í¥é”õ…ÉÌ¹‰…Ñ¡}É½ÕÁÌ°Í…µÁ±•ÈõÙ…±¥‘}Í…µÁ±•È°½±±…Ñ•}™¸õ½±±…Ñ•}É½ÕÁÌ¤((€€€½ÁÑ¥´€ôÑ½É ¹½ÁÑ¥´¹‘…µ\¡µ½‘•°¹Á…É…µ•Ñ•ÉÌ ¤°±Èõ…ÉÌ¹±È°Ý•¥¡Ñ}‘•…äõ…ÉÌ¹Ý•¥¡Ñ}‘•…ä¤((€€€ÍÑ•ÁÍ}Á•É}•Á½ €ô±•¸¡ÑÉ}±¤(€€€½µÁÕÑ•‘}Ñ½Ñ…±}ÍÑ•ÁÌ€ô…ÉÌ¹•Á½¡Ì€¨ÍÑ•ÁÍ}Á•É}•Á½ ((€€€¥˜…ÉÌ¹µ…á}ÍÑ•ÁÌ€ø€Àè(€€€€€€€Ñ½Ñ…±}ÍÑ•ÁÌ€ô…ÉÌ¹µ…á}ÍÑ•ÁÌ(((€€€€€€€¥˜Ñ½Ñ…±}ÍÑ•ÁÌ€ð½µÁÕÑ•‘}Ñ½Ñ…±}ÍÑ•ÁÌè(€€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜‰m%¹™½t=Ù•ÉÉ¥‘¥¹œÑ½Ñ…±}ÍÑ•ÁÌ™É½´í½µÁÕÑ•‘}Ñ½Ñ…±}ÍÑ•ÁÍôÑ¼íÑ½Ñ…±}ÍÑ•ÁÍô‘Õ”Ñ¼€´µµ…á}ÍÑ•ÁÌˆ¤(€€€•±Í”è(€€€€€€€Ñ½Ñ…±}ÍÑ•ÁÌ€ô½µÁÕÑ•‘}Ñ½Ñ…±}ÍÑ•ÁÌ((€€€Ý…Éµ}ÍÑ•ÁÌ€ô¥¹Ð¡Ñ½Ñ…±}ÍÑ•ÁÌ€¨…ÉÌ¹Ý…ÉµÕÁ}É…Ñ¥¼¤(€€€Í¡•€ô•Ñ}±¥¹•…É}Í¡•‘Õ±•}Ý¥Ñ¡}Ý…ÉµÕÀ¡½ÁÑ¥´°Ý…Éµ}ÍÑ•ÁÌ°Ñ½Ñ…±}ÍÑ•ÁÌ¤(€€€Í…±•È€ôÑ½É ¹…µÀ¹É…‘M…±•È Õ‘„œ°•¹…‰±•õQÉÕ”¤((€€€‰•ÍÑ}¹‘œ€ô€´Ä¸À(€€€‰•ÍÑ}‘¥È€ô½Ì¹Á…Ñ ¹©½¥¸¡…ÉÌ¹½ÕÑ}‘¥È°€‰‰•ÍÐˆ¤ì±…ÍÑ}‘¥È€ô½Ì¹Á…Ñ ¹©½¥¸¡…ÉÌ¹½ÕÑ}‘¥È°€‰±…ÍÐˆ¤(€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€½Ì¹µ…­•‘¥ÉÌ¡‰•ÍÑ}‘¥È°•á¥ÍÑ}½¬õQÉÕ”¤ì½Ì¹µ…­•‘¥ÉÌ¡±…ÍÑ}‘¥È°•á¥ÍÑ}½¬õQÉÕ”¤((€€€ÍÑ•ÁÍ}‘½¹”€ô€Àì•µ…}ÍÑ•Á}Í•Œ€ô9½¹”ìÍÑ•À€ô€ÀìÉÕ¹}±½ÍÌ€ô€À¸À(€€€ÍÑ…ÉÑ}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤((€€€ÍÑ½Á}ÑÉ…¥¹¥¹œ€ô…±Í”(((€€€M=I}1=}YId€ôµ…à Ä°…ÉÌ¹±½}ÍÑ•ÁÌ€¼¼€È¤((€€€™½È•À¥¸É…¹” Ä°…ÉÌ¹•Á½¡Ì¬Ä¤è(€€€€€€€¥˜ÍÑ½Á}ÑÉ…¥¹¥¹œè‰É•…¬(((€€€€€€€µ½‘•°¹ÑÉ…¥¸ ¤(€€€€€€€ÑÉ…¥¹}Í…µÁ±•È¹Í•Ñ}•Á½ ¡•À¤(€€€€€€€•Á}ÍÑ…ÉÐ€ôÑ¥µ”¹Ñ¥µ” ¤((€€€€€€€™½ÈD°°d°±•¹Ì°É½ÕÁ}±…‰•±Ì¥¸ÑÉ}±è(€€€€€€€€€€€¥˜…ÉÌ¹µ…á}ÍÑ•ÁÌ€ø€À…¹ÍÑ•À€øô…ÉÌ¹µ…á}ÍÑ•ÁÌè(€€€€€€€€€€€€€€€ÍÑ½Á}ÑÉ…¥¹¥¹œ€ôQÉÕ”(€€€€€€€€€€€€€€€‰É•…¬(€€€€€€€€€€€ÐÀ€ôÑ¥µ”¹Ñ¥µ” ¤((€€€€€€€€€€€•¹Œ€ô•¹½‘•}Á…¥ÉÍ}ÍÁ±¥Ñ}ÑÉÕ¹Œ¡Ñ½¬°D°°Å}µ…àõ…ÉÌ¹Å}µ…à°‘}µ…àõ…ÉÌ¹‘}µ…à°‘•Ù¥”õ‘•Ù¥”¤(€€€€€€€€€€€Ý¥Ñ Ñ½É ¹…µÀ¹…ÕÑ½…ÍÐ Õ‘„œ°‘ÑåÁ”õÑ½É ¹‰™±½…ÐÄØ¤è(€€€€€€€€€€€€€€€½ÕÐ€ôµ½‘•°¡•¹l‰¥¹ÁÕÑ}¥‘Ì‰t°•¹l‰…ÑÑ•¹Ñ¥½¹}µ…Í¬‰t¹±½¹œ ¤°µ½‘”ô‰É•É…¹¬ˆ¤(€€€€€€€€€€€€€€€Í½É•Ì€ôÑ½É ¹¹…¹}Ñ½}¹Õ´¡½ÕÑl‰Í½É•Ì‰t¹ÍÅÕ••é” ´Ä¤°¹…¸ô´Å”ä¤(€€€€€€€€€€€€€€€±½ÍÌ€ôÉ½ÕÁÝ¥Í•}±½ÍÌ¡Í½É•Ì°d¹Ñ¼¡‘•Ù¥”¤°±•¹Ì°É½ÕÁ}±…‰•±Ì¹Ñ¼¡‘•Ù¥”¤°(€€€€€€€€€€€€€€€€€€€€€É…¹­}±½ÍÌõ…ÉÌ¹É…¹­}±½ÍÌ°µ…É¥¸õ…ÉÌ¹µ…É¥¸°(€€€€€€€€€€€€€€€€€€€€€…±Á¡„õ…ÉÌ¹…±Á¡„°…µµ„õ…ÉÌ¹…µµ„°(€€€€€€€€€€€€€€€€€€€€€‰•Ñ…}•¹Ðõ…ÉÌ¹‰•Ñ…}•¹Ð°Põ…ÉÌ¹Ñ•µÀ°(€€€€€€€€€€€€€€€€€€€€€µ…á}Á…¥ÉÌõ…ÉÌ¹µ…á}Á…¥ÉÌ°‘•Ù¥”õ‘•Ù¥”°(€€€€€€€€€€€€€€€€€€€€€‰•Ñ…}%Iõ…ÉÌ¹‰•Ñ…}%I¤(€€€€€€€€€€€Í…±•È¹Í…±”¡±½ÍÌ¤¹‰…­Ý…É ¤(((€€€€€€€€€€€Í…±•È¹Õ¹Í…±•|¡½ÁÑ¥´¤((€€€€€€€€€€€É…‘}¹½É´€ôÑ½É ¹Ñ•¹Í½È À¸À°‘•Ù¥”õ‘•Ù¥”¤(€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è((€€€€€€€€€€€€€€€É…‘Ì€ômÀ¹É…¹‘•Ñ…  ¤™½ÈÀ¥¸µ½‘•°¹Á…É…µ•Ñ•ÉÌ ¤¥˜À¹É…¥Ì¹½Ð9½¹•t(€€€€€€€€€€€€€€€¥˜É…‘Ìè(€€€€€€€€€€€€€€€€€€€É…‘}¹½É´€ôÑ½É ¹¹½É´¡Ñ½É ¹ÍÑ…¬¡mÑ½É ¹¹½É´¡œ°€È¤™½Èœ¥¸É…‘Ít¤°€È¤(((€€€€€€€€€€€¹¸¹ÕÑ¥±Ì¹±¥Á}É…‘}¹½Éµ|¡µ½‘•°¹Á…É…µ•Ñ•ÉÌ ¤°…ÉÌ¹É…‘}±¥À¤((€€€€€€€€€€€Í…±•È¹ÍÑ•À¡½ÁÑ¥´¤ìÍ…±•È¹ÕÁ‘…Ñ” ¤ìÍ¡•¹ÍÑ•À ¤(€€€€€€€€€€€½ÁÑ¥´¹é•É½}É…¡Í•Ñ}Ñ½}¹½¹”õQÉÕ”¤((€€€€€€€€€€€±½ÍÍ}‘•Ñ…¡•€ô™±½…Ð¡±½ÍÌ¹‘•Ñ…  ¤¤(€€€€€€€€€€€ÉÕ¹}±½ÍÌ€¬ô±½ÍÍ}‘•Ñ…¡•ìÍÑ•À€¬ô€ÄìÍÑ•ÁÍ}‘½¹”€¬ô€Ä(((€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤…¹ÍÑ•À€”M=I}1=}YId€ôô€Àè(€€€€€€€€€€€€€€€}±½}É…Ý}Í½É•Ì¡D°°d°±•¹Ì°Í½É•Ì°ÍÑ•À°•À¤((€€€€€€€€€€€ÍÑ•Á}Í•Œ€ôÑ¥µ”¹Ñ¥µ” ¤€´ÐÀ(€€€€€€€€€€€¥˜•µ…}ÍÑ•Á}Í•Œ¥Ì9½¹”è•µ…}ÍÑ•Á}Í•Œ€ôÍÑ•Á}Í•Œ(€€€€€€€€€€€•±Í”è•µ…}ÍÑ•Á}Í•Œ€ô€À¸ä©•µ…}ÍÑ•Á}Í•Œ€¬€À¸Ä©ÍÑ•Á}Í•Œ(€€€€€€€€€€€•Ñ…}Í•Œ€ôµ…à À¸À°€¡Ñ½Ñ…±}ÍÑ•ÁÌµÍÑ•ÁÍ}‘½¹”¤€¨•µ…}ÍÑ•Á}Í•Œ¤((€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤…¹ÍÑ•À€”…ÉÌ¹±½}ÍÑ•ÁÌ€ôô€Àè(€€€€€€€€€€€€€€€…Ù}±•¸°µ…á}•™™}±•¸°Á…‘}±•¸€ôÍ•Å}±•¹}ÍÑ…ÑÌ¡•¹l‰…ÑÑ•¹Ñ¥½¹}µ…Í¬‰t¤(€€€€€€€€€€€€€€€…Ù}Á½Ì°…Ù}¹•œ°…Ù}Á…¥ÉÌ€ô‰…Ñ¡}ÍÑ…ÑÌ¡d°±•¹Ì°…ÉÌ¹µ…á}Á…¥ÉÌ¤(€€€€€€€€€€€€€€€¥˜Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤è(€€€€€€€€€€€€€€€€€€€…±±½Œ€ôÑ½É ¹Õ‘„¹µ•µ½Éå}…±±½…Ñ•¡‘•Ù¥”¤€¼€ ÄÀÈÐ¨¨Ì¤(€€€€€€€€€€€€€€€€€€€É•Í•ÉÙ•€ôÑ½É ¹Õ‘„¹µ•µ½Éå}É•Í•ÉÙ•¡‘•Ù¥”¤€¼€ ÄÀÈÐ¨¨Ì¤(€€€€€€€€€€€€€€€€€€€µ•µ}¥¹™¼€ô˜‰AUí±½…±}É…¹­ôµ•´…±±½Œ½É•ÍØèí…±±½Œè¸Å™õ½íÉ•Í•ÉÙ•è¸Å™õˆ(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€…±±½Œ€ôÉ•Í•ÉÙ•€ô€À¸À(€€€€€€€€€€€€€€€€€€€µ•µ}¥¹™¼€ô€‰ATµ•´è8½ˆ(€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}±È€ôÍ¡•¹•Ñ}±…ÍÑ}±È ¥lÁt((€€€€€€€€€€€€€€€É…‘}¹½Éµ}ÁÔ€ô™±½…Ð¡É…‘}¹½É´¹¥Ñ•´ ¤¤¥˜¥Í}µ…¥¸¡É…¹¬¤•±Í”€À¸À((€€€€€€€€€€€€€€€ÁÉ¥¹Ð (€€€€€€€€€€€€€€€€€€€˜‰m5…¥¹t•Àí•ÁôÍÑ•ÀíÍÑ•Áô½íÑ½Ñ…±}ÍÑ•ÁÍô€ˆ(€€€€€€€€€€€€€€€€€€€˜‰±½ÍÌõíÉÕ¹}±½ÍÌ½…ÉÌ¹±½}ÍÑ•ÁÌè¸Ñ™ô±ÈõíÕÉÉ•¹Ñ}±Èè¸É•ô€ˆ(€€€€€€€€€€€€€€€€€€€˜‰É…‘}¹½É´õíÉ…‘}¹½Éµ}ÁÔè¸Ñ™ô€ˆ(€€€€€€€€€€€€€€€€€€€˜‰ðÍ•Å}±•¸…Ùœ½µ…à½Á…õí…Ù}±•¸è¸Å™ô½íµ…á}•™™}±•¹ô½íÁ…‘}±•¹ô€ˆ(€€€€€€€€€€€€€€€€€€€˜‰ðÉÀ…ÙœÁ½Ì½¹•œõí…Ù}Á½Ìè¸Å™ô½í…Ù}¹•œè¸Å™ôÁ…¥ÉÍùí…Ù}Á…¥ÉÌè¸Á™ô€ˆ(€€€€€€€€€€€€€€€€€€€˜‰ðÍÑ•ÀõíÍÑ•Á}Í•Œè¸É™õÌ•µ„õí•µ…}ÍÑ•Á}Í•Œè¸É™õÌQõí™µÑ}Ñ¥µ”¡•Ñ…}Í•Œ¥ô€ˆ(€€€€€€€€€€€€€€€€€€€˜‰ðíµ•µ}¥¹™½ôˆ(€€€€€€€€€€€€€€€€¤(((€€€€€€€€€€€€€€€±½•È¹±½}ÍÑ•À (€€€€€€€€€€€€€€€€€€€•Á½ õ•À°(€€€€€€€€€€€€€€€€€€€±½‰…±}ÍÑ•ÀõÍÑ•À°(€€€€€€€€€€€€€€€€€€€±½ÍÍ}…ÙœõÉÕ¹}±½ÍÌ½…ÉÌ¹±½}ÍÑ•ÁÌ°(€€€€€€€€€€€€€€€€€€€±ÈõÕÉÉ•¹Ñ}±È°(€€€€€€€€€€€€€€€€€€€É…‘}¹½É´õÉ…‘}¹½Éµ}ÁÔ°(€€€€€€€€€€€€€€€€€€€Í•Å}±•¹}…Ùœõ…Ù}±•¸°(€€€€€€€€€€€€€€€€€€€Í•Å}±•¹}µ…àõµ…á}•™™}±•¸°(€€€€€€€€€€€€€€€€€€€Á…‘}±•¸õÁ…‘}±•¸°(€€€€€€€€€€€€€€€€€€€…Ù}Á½Ìõ…Ù}Á½Ì°(€€€€€€€€€€€€€€€€€€€…Ù}¹•œõ…Ù}¹•œ°(€€€€€€€€€€€€€€€€€€€…Ù}Á…¥ÉÌõ…Ù}Á…¥ÉÌ°(€€€€€€€€€€€€€€€€€€€ÍÑ•Á}Í•ŒõÍÑ•Á}Í•Œ°(€€€€€€€€€€€€€€€€€€€•µ…}ÍÑ•Á}Í•Œõ•µ…}ÍÑ•Á}Í•Œ°(€€€€€€€€€€€€€€€€€€€•Ñ…}Í•Œõ•Ñ…}Í•Œ°(€€€€€€€€€€€€€€€€€€€ÁÕ}µ•µ}…±±½}ˆõ…±±½Œ°(€€€€€€€€€€€€€€€€€€€ÁÕ}µ•µ}É•Í•ÉÙ•‘}ˆõÉ•Í•ÉÙ•°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€±½•È¹Á±½Ñ}ÕÉÙ•Ì ¤(€€€€€€€€€€€€€€€ÉÕ¹}±½ÍÌ€ô€À¸À((€€€€€€€€€€€€Œ‘•°•¹Œ°½ÕÐ°Í½É•Ì(€€€€€€€€€€€€Œ¥˜Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤èÑ½É ¹Õ‘„¹•µÁÑå}…¡” ¤(€€€€€€€€€€€¥˜…ÉÌ¹µ…á}ÍÑ•ÁÌ€ø€À…¹ÍÑ•À€øô…ÉÌ¹µ…á}ÍÑ•ÁÌè(€€€€€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜‰mMÑ½ÁtI•…¡•µ…á}ÍÑ•ÁÌõí…ÉÌ¹µ…á}ÍÑ•ÁÍô¸á¥Ñ¥¹œÑÉ…¥¹¥¹œ±½½À¸ˆ¤(€€€€€€€€€€€€€€€ÍÑ½Á}ÑÉ…¥¹¥¹œ€ôQÉÕ”(€€€€€€€€€€€€€€€‰É•…¬((€€€€€€€•Á}Ñ¥µ”€ôÑ¥µ”¹Ñ¥µ” ¤€´•Á}ÍÑ…ÉÐ(€€€€€€€‰…ÉÉ¥•È ¤(€€€€€€€Ù…±¥‘}Í…µÁ±•È¹Í•Ñ}•Á½ ¡•À¤(€€€€€€€µ•ÑÉ¥Ì€ô•Ù…±Õ…Ñ”¡µ½‘•°°Ñ½¬°Ù…}±°‘•Ù¥”°(€€€€€€€€€€€€€€€€€€€€€€€€€€ÕÑ½™˜ôÄÀ°Põ…ÉÌ¹Ñ•µÀ°Å}µ…àõ…ÉÌ¹Å}µ…à°‘}µ…àõ…ÉÌ¹‘}µ…à°É…¹¬õÉ…¹¬¤((€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€€€€€ÁÉ¥¹Ð¡˜‰q¸ôôøÁ½ í•Áô½µÁ±•Ñ•¥¸í™µÑ}Ñ¥µ”¡•Á}Ñ¥µ”¥ôˆ¤(€€€€€€€€€€€ÁÉ¥¹Ð ‰Y…±¥‘…Ñ¥½¸5•ÑÉ¥Ìèˆ¤(€€€€€€€€€€€¥˜€‰5II ÄÀˆ¥¸µ•ÑÉ¥Ìè(€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€5II ÄÀèíµ•ÑÉ¥Íl5II ÄÀtè¸Ñ™ôðA Ôèíµ•ÑÉ¥ÍlA Ôtè¸Ñ™ôðA ÄÀèíµ•ÑÉ¥ÍlA ÄÀtè¸Ñ™ôðA½ÍÉ½ÕÁÌèíµ•ÑÉ¥ÍlA½ÍÉ½ÕÁÌuôˆ¤(€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€Q½ÀÅ}Œèíµ•ÑÉ¥ÍlQ½ÀÅ}Œtè¸Ñ™ôð5¥¹}A½Í}I…¹¬èíµ•ÑÉ¥Íl5¥¹}A½Í}I…¹¬tè¸É™ôð5…á}9•}I…¹¬èíµ•ÑÉ¥Íl5…á}9•}I…¹¬tè¸É™ôðI…¹­}…Àèíµ•ÑÉ¥ÍlI…¹­}…Àtè¸É™ôð-1}¥Øèíµ•ÑÉ¥Íl-1}¥Øtè¸Ñ™ôˆ¤(€€€€€€€€€€€¥˜€‰9•É½ÕÁÌˆ¥¸µ•ÑÉ¥Ìè(€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜ˆ€9•}-0¡µ•…¸¤èíµ•ÑÉ¥Íl9•}-0¡µ•…¸¤tè¸Ñ™ôð9•}5…áM½É”¡µ•…¸¤èíµ•ÑÉ¥Íl9•}5…áM½É”¡µ•…¸¤tè¸Ñ™ôð9•É½ÕÁÌèíµ•ÑÉ¥Íl9•É½ÕÁÌuôˆ¤(€€€€€€€€€€€ÁÉ¥¹Ð ˆ´ˆ¨ØÀ¤(((€€€€€€€€€€€±½•È¹±½}•Á½ ¡•À°•Á}Ñ¥µ”°µ•ÑÉ¥Ì¤(€€€€€€€€€€€±½•È¹Á±½Ñ}ÕÉÙ•Ì ¤((€€€€€€€€€€€Í½É•}™½É}‰•ÍÐ€ôµ•ÑÉ¥Ì¹•Ð ‰5II ÄÀˆ°€À¸À¤((€€€€€€€€€€€¥˜Í½É•}™½É}‰•ÍÐ€ø‰•ÍÑ}¹‘œè(€€€€€€€€€€€€€€€‰•ÍÑ}¹‘œ€ôÍ½É•}™½É}‰•ÍÐ(€€€€€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€€€€€€€€€€€€€¥˜…ÉÌ¹ÕÍ•}±½É„è((€€€€€€€€€€€€€€€€€€€€€€€µ½‘•°¹µ½‘Õ±”¹Í…Ù•}ÁÉ•ÑÉ…¥¹•¡‰•ÍÑ}‘¥È¤(€€€€€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€€€€€µ½‘•°¹µ½‘Õ±”¹Í…Ù•}ÁÉ•ÑÉ…¥¹•¡‰•ÍÑ}‘¥È¤(€€€€€€€€€€€€€€€€€€€Ñ½¬¹Í…Ù•}ÁÉ•ÑÉ…¥¹•¡‰•ÍÑ}‘¥È¤(€€€€€€€€€€€€€€€€€€€Ñ½É ¹Í…Ù”¡ì‰•Á½ ˆè•À°€‰‰•ÍÑ}5II ÄÀˆè‰•ÍÑ}¹‘ô°½Ì¹Á…Ñ ¹©½¥¸¡‰•ÍÑ}‘¥È°€‰ÑÉ…¥¹•É}ÍÑ…Ñ”¹ÁÐˆ¤¤(€€€€€€€€€€€€€€€€€€€ÁÉ¥¹Ð¡˜‰mÍ…Ù•t‰•ÍÐ€´øí‰•ÍÑ}‘¥Éô€¡5II ÄÀèí‰•ÍÑ}¹‘œè¸Ñ™ô¤ˆ¤(((€€€€€€€€€€€¥˜¥Í}µ…¥¸¡É…¹¬¤è(€€€€€€€€€€€€€€€¥˜…ÉÌ¹ÕÍ•}±½É„è((€€€€€€€€€€€€€€€€€€€µ½‘•°¹µ½‘Õ±”¹Í…Ù•}ÁÉ•ÑÉ…¥¹•¡±…ÍÑ}‘¥È¤(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€µ½‘•°¹µ½‘Õ±”¹Í…Ù•}ÁÉ•ÑÉ…¥¹•¡±…ÍÑ}‘¥È¤(€€€€€€€€€€€€€€€Ñ½¬¹Í…Ù•}ÁÉ•ÑÉ…¥¹•¡±…ÍÑ}‘¥È¤(€€€€€€€€€€€€€€€Ñ½É ¹Í…Ù”¡ì‰•Á½ ˆè•À°€‰5II ÄÀˆèÍ½É•}™½É}‰•ÍÑô°½Ì¹Á…Ñ ¹©½¥¸¡±…ÍÑ}‘¥È°€‰ÑÉ…¥¹•É}ÍÑ…Ñ”¹ÁÐˆ¤¤()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤