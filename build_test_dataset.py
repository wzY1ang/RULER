import json
import argparse
import os
from collections import defaultdict

def clean_text(text):
    if not text: return ""
    return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()

def load_law_corpus(corpus_path):
    print(f"Loading law corpus from {corpus_path}...")
    law_dict = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                did = str(doc.get('text_id', doc.get('doc_id', '')))
                text = doc.get('text') or ''
                name = doc.get('name') or ''
                full_text = f"{name} {text}".strip() if name or text else ""
                law_dict[did] = clean_text(full_text)
    return law_dict

def load_cases(json_path):
    print(f"Loading cases from {json_path}...")
    data = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            data = content if isinstance(content, list) else []
    except:
        with open(json_path, 'r', encoding='utf-8') as f:
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    case_dict = {}
    for item in data:
        qid = str(item.get('text_id', item.get('qid', '')))
        text = item.get('text', item.get('query', ''))
        la = item.get('la', item.get('positives', []))
        if isinstance(la, str):
            la = [x.strip() for x in la.split(',') if x.strip()]
        la = [str(x) for x in la]
        case_dict[qid] = {'text': clean_text(text), 'la': set(la)}
    return case_dict

def load_rank_tsv(rank_path):
    print(f"Loading rankings from {rank_path}...")
    rank_dict = defaultdict(list)
    with open(rank_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            qid, did = parts[0], parts[1]
            rank_dict[qid].append(did)
    return rank_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_json', required=True, help="Test ground-truth file")
    parser.add_argument('--law_corpus', required=True, help="Article corpus")
    parser.add_argument('--rank_test', required=True, help="Test retrieval TSV")
    parser.add_argument('--out_file', required=True, help="Output TSV path")
    parser.add_argument('--topk', type=int, default=50, help="Candidates retained per query")
    args = parser.parse_args()


    law_corpus = load_law_corpus(args.law_corpus)
    cases = load_cases(args.test_json)
    rank_dict = load_rank_tsv(args.rank_test)

    print(f"Constructing TEST dataset (Top-{args.topk})...")

    with open(args.out_file, 'w', encoding='utf-8') as out_f:
        out_f.write("query_id\tquery\tcand_id\tcand_text\tgroup_id\tcand_label\tgroup_label\n")

        count = 0
        for qid, case in cases.items():
            if qid not in rank_dict: continue


            cand_ids = rank_dict[qid][:args.topk]
            pos_set = case['la']
            query_text = case['text']


            group_has_pos = any(did in pos_set for did in cand_ids)
            group_label = 1 if group_has_pos else 0


            for cand_id in cand_ids:
                cand_text = law_corpus.get(cand_id, "[MISSING]")
                cand_label = 1 if cand_id in pos_set else 0


                out_f.write(f"{qid}\t{query_text}\t{cand_id}\t{cand_text}\t{qid}\t{cand_label}\t{group_label}\n")

            count += 1

    print(f"Test dataset built: {count} queries processed.")

if __name__ == '__main__':
    main()
