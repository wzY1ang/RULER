import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import roc_curve

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data(tsv_file, score_file):

    print(f"Loading labels from {tsv_file}...")
    try:

        df = pd.read_csv(tsv_file, sep='\t')


        if 'cand_label' in df.columns:
            labels = df['cand_label'].values
            qids = df['group_id'].values if 'group_id' in df.columns else None
        elif 'label' in df.columns:
            labels = df['label'].values
            qids = df['query_id'].values if 'query_id' in df.columns else None
        else:

            df = pd.read_csv(tsv_file, sep='\t', header=None)
            if df.shape[1] >= 6:
                labels = df.iloc[:, 5].values
                qids = df.iloc[:, 2].values
            else:
                labels = df.iloc[:, -1].values
                qids = df.iloc[:, 0].values

    except Exception as e:
        print(f"Error loading TSV: {e}")
        return None, None, None


    print(f"Loading scores from {score_file}...")
    scores_list = []
    try:
        with open(score_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue


                parts = line.split()


                try:
                    score = float(parts[-1])
                    scores_list.append(score)
                except ValueError:

                    print(f"Warning: could not parse line {i+1}: {line}. Skipping.")
                    continue

        scores = np.array(scores_list)
        print(f"   Successfully loaded {len(scores)} scores.")

    except Exception as e:
        print(f"Error loading scores: {e}")
        return None, None, None


    min_len = min(len(labels), len(scores))
    if len(labels) != len(scores):
        print(f"Warning: label/score length mismatch ({len(labels)} vs. {len(scores)}); truncating to {min_len}.")


    if qids is None or len(qids) < min_len:
        qids = np.arange(min_len)

    return qids[:min_len], labels[:min_len], scores[:min_len]

def calc_overlap_ratio(qids, labels, scores):
    """
    Compute the fraction of queries where max(negative) >= min(positive).
    """

    df = pd.DataFrame({'qid': qids, 'label': labels, 'score': scores})

    overlap_count = 0
    total_valid_groups = 0


    for qid, group in df.groupby('qid'):
        pos_scores = group[group['label'] == 1]['score'].values
        neg_scores = group[group['label'] == 0]['score'].values


        if len(pos_scores) > 0 and len(neg_scores) > 0:
            total_valid_groups += 1

            if np.max(neg_scores) >= np.min(pos_scores):
                overlap_count += 1

    ratio = overlap_count / total_valid_groups if total_valid_groups > 0 else 0.0
    return ratio

def calc_calibrated_metrics(labels, raw_scores):
    """
    Select a threshold with ROC analysis and compute the calibrated score gap.
    """

    fpr, tpr, thresholds = roc_curve(labels, raw_scores)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]


    optimal_bias = -best_threshold


    calibrated_logits = raw_scores + optimal_bias
    calibrated_probs = sigmoid(calibrated_logits)


    pos_probs = calibrated_probs[labels == 1]
    neg_probs = calibrated_probs[labels == 0]

    gap = 0.0
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        gap = np.mean(pos_probs) - np.mean(neg_probs)

    return optimal_bias, gap, np.mean(pos_probs), np.mean(neg_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="Test TSV containing labels and group IDs")
    parser.add_argument("--score", required=True, help="Model score file")
    parser.add_argument("--model_name", default="Model", help="Model name used in the report")
    args = parser.parse_args()


    qids, labels, scores = load_data(args.tsv, args.score)
    if labels is None or scores is None: return


    overlap_ratio = calc_overlap_ratio(qids, labels, scores)


    bias, calib_gap, pos_avg, neg_avg = calc_calibrated_metrics(labels, scores)


    print("\n" + "="*50)
    print(f"Evaluation report: {args.model_name}")
    print("="*50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 40)
    print(f"Calibrated Pos Avg        | {pos_avg:.4f}")
    print(f"Calibrated Neg Avg        | {neg_avg:.4f}")
    print(f"Optimal Bias              | {bias:.4f}")
    print("-" * 40)
    print(f"Calibrated Score Gap      | {calib_gap:.4f}  (Discrimination)")
    print(f"Score Overlap Ratio       | {overlap_ratio:.4f}  (Safety)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
