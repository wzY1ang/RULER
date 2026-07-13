import json
import argparse
import os
import random
from collections import defaultdict


TRAIN_GROUP_SIZE = 10
NUM_MIXED_GROUPS = 3
NUM_NEG_GROUPS =  2

# ============================

# ============================

def clean_text(text):
    if not text: return ""
    return text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()

def load_law_corpus(corpus_path):
    print(f"Loading law corpus from {corpus_path}...")
    law_dict = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    doc = json.loads(line)
                    did = str(doc.get('text_id', doc.get('doc_id', '')))
                    text = doc.get('text') or ''
                    name = doc.get('name') or ''
                    full_text = f"{name} {text}".strip() if name or text else ""
                    law_dict[did] = clean_text(full_text)
                except:
                    continue
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
                    try:
                        data.append(json.loads(line))
                    except:
                        continue

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

# ============================

# ============================

# ============================

# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', required=True, help="Training ground-truth file")
    parser.add_argument('--law_corpus', required=True, help="Article corpus")
    parser.add_argument('--rank_train', required=True, help="Training retrieval TSV")
    parser.add_argument('--out_file', required=True, help="Output TSV path")
    parser.add_argument('--sample_n', type=int, default=None, help="Optional random sample size")

    args = parser.parse_args()


    law_corpus = load_law_corpus(args.law_corpus)
    all_doc_ids = list(law_corpus.keys())


    print("\nInspecting the training query file...")
    try:
        with open(args.train_json, 'r', encoding='utf-8') as f:

            first_line = f.readline()
            if first_line.strip():
                try:
                    sample_json = json.loads(first_line)
                    print(f"First sample keys: {list(sample_json.keys())}")
                    print(f"First sample preview: {json.dumps(sample_json, ensure_ascii=False)[:200]}...")
                except:

                    f.seek(0)
                    content = json.load(f)
                    if isinstance(content, list) and len(content) > 0:
                        print(f"First sample keys: {list(content[0].keys())}")
    except Exception as e:
        print(f"Could not inspect the training query file: {e}")
    print("-" * 30)

    cases = load_cases(args.train_json)
    rank_dict = load_rank_tsv(args.rank_train)


    case_ids = set(cases.keys())
    rank_ids = set(rank_dict.keys())
    intersection = case_ids.intersection(rank_ids)

    print(f"Train Cases IDs: {len(case_ids)}")
    print(f"Rank File IDs:   {len(rank_ids)}")
    print(f"Matched query IDs: {len(intersection)}")

    if len(intersection) == 0:
        return

    process_qids = list(intersection)
    if args.sample_n is not None and args.sample_n > 0:
         process_qids = random.sample(process_qids, args.sample_n)

    print("\nConstructing TRAIN dataset (3 Mixed + 2 Neg)...")

    with open(args.out_file, 'w', encoding='utf-8') as out_f:
        out_f.write("query_id\tquery\tcand_id\tcand_text\tgroup_id\tcand_label\tgroup_label\n")

        count = 0
        error_count = 0
        debug_skips = 0

        for qid in process_qids:
            case = cases[qid]
            cand_ids = rank_dict[qid][:50]
            pos_set = case['la']


            if debug_skips < 5:
                is_skip = False
                reason = ""
                if not cand_ids:
                    is_skip = True; reason = "candidate set is empty"
                elif not pos_set:
                    is_skip = True; reason = "positive set is empty; neither 'la' nor 'positives' was parsed"

                if is_skip:
                    print(f"Skipping QID {qid}: {reason}")
                    debug_skips += 1
            # ------------------------------------

            if not cand_ids: continue

            query_text = case['text']
            hard_negs = [did for did in cand_ids if did not in pos_set]
            pos_list = list(pos_set)

            if not pos_list: continue




            def get_negs_safely(needed_count, current_group_pos):
                nonlocal error_count
                if needed_count <= 0: return []
                try:
                    exclude = set(current_group_pos)
                    valid_hard = [h for h in hard_negs if h not in exclude]
                    return random.sample(valid_hard, needed_count)
                except ValueError:
                    if error_count < 1: print(f"Insufficient candidates for QID {qid}")
                    error_count += 1
                    selected = list(valid_hard)
                    shortage = needed_count - len(selected)
                    full_exclude = pos_set.union(set(selected))
                    extras = []
                    while len(extras) < shortage:
                        r_id = random.choice(all_doc_ids)
                        if r_id not in full_exclude and r_id not in extras:
                            extras.append(r_id)
                    return selected + extras
            # -------------------------

            # A. Mixed Groups
            group_pos_buckets = [set() for _ in range(NUM_MIXED_GROUPS)]
            for pid in pos_list:
                group_pos_buckets[random.randint(0, NUM_MIXED_GROUPS - 1)].add(pid)
            for i in range(NUM_MIXED_GROUPS):
                if not group_pos_buckets[i]: group_pos_buckets[i].add(random.choice(pos_list))

            max_pos = TRAIN_GROUP_SIZE - 1
            for i in range(NUM_MIXED_GROUPS):
                curr = len(group_pos_buckets[i])
                if curr < max_pos:
                    cap = max_pos - curr
                    add_n = random.randint(0, min(cap, len(pos_list)))
                    if add_n: group_pos_buckets[i].update(random.sample(pos_list, add_n))

            groups_to_write = []
            for i in range(NUM_MIXED_GROUPS):
                curr_pos = list(group_pos_buckets[i])
                if len(curr_pos) > max_pos:
                    random.shuffle(curr_pos)
                    curr_pos = curr_pos[:max_pos]
                num_negs = TRAIN_GROUP_SIZE - len(curr_pos)
                batch_negs = get_negs_safely(num_negs, curr_pos)
                groups_to_write.append({"group_id": f"{qid}_mix_{i}", "docs": curr_pos + batch_negs, "group_label": 1})

            # B. Neg Groups
            for i in range(NUM_NEG_GROUPS):
                batch_negs = get_negs_safely(TRAIN_GROUP_SIZE, [])
                groups_to_write.append({"group_id": f"{qid}_neg_{i}", "docs": batch_negs, "group_label": 0})


            for grp in groups_to_write:
                g_id = grp['group_id']
                g_lbl = grp['group_label']
                for cid in grp['docs']:
                    ctext = law_corpus.get(cid, "[MISSING]")
                    clbl = 1 if cid in pos_set else 0
                    out_f.write(f"{qid}\t{query_text}\t{cid}\t{ctext}\t{g_id}\t{clbl}\t{g_lbl}\n")

            count += 1

    print(f"Training dataset built: {count} queries processed.")

if __name__ == '__main__':
    main()
