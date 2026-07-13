import argparse
import json
from collections import Counter, defaultdict


def load_json_or_jsonl(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            content = json.load(f)
            return content if isinstance(content, list) else [content]
        except Exception:
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
            return data


def normalize_article_ids(item):
    raw_articles = item.get('la') or item.get('article') or []
    if not isinstance(raw_articles, list):
        raw_articles = [raw_articles]

    article_ids = set()
    for art in raw_articles:
        try:
            article_ids.add(int(float(art)))
        except Exception:
            pass

    return sorted(article_ids)


def split_general_specific(article_ids):
    generals = sorted([x for x in article_ids if x <= 101])
    specifics = sorted([x for x in article_ids if x > 101])
    return generals, specifics


def filter_and_analyze(input_path,
                       output_path,
                       stats_output_path=None,
                       min_labels=3,
                       top_k=20):
    """
    Filter gold data and report label-combination statistics.

    Retained samples have at least ``min_labels`` labels, including one
    general article (<= 101) and one specific article (> 101).
    """

    print("Filtering gold data and analyzing label combinations")
    print(f"Criteria: at least {min_labels} labels, with general (<=101) and specific (>101) articles")

    data = load_json_or_jsonl(input_path)

    stats = {
        "total": 0,
        "kept": 0,
        "discarded": 0,
        "discard_reason": {
            "too_few_labels": 0,
            "missing_specific": 0,
            "missing_general": 0,
            "parse_error": 0
        }
    }

    kept_data = []


    all_combo_counter = Counter()
    all_specific_general_pattern_counter = Counter()


    kept_combo_counter = Counter()
    kept_specific_general_pattern_counter = Counter()


    discarded_only_specific_counter = Counter()
    discarded_only_general_counter = Counter()
    discarded_too_few_counter = Counter()


    kept_specific_to_general_sets = defaultdict(Counter)

    for item in data:
        try:
            stats["total"] += 1

            article_ids = normalize_article_ids(item)
            combo = tuple(article_ids)

            generals, specifics = split_general_specific(article_ids)


            if combo:
                all_combo_counter[combo] += 1

            for sp in specifics:
                all_specific_general_pattern_counter[(sp, tuple(generals))] += 1


            if len(article_ids) < min_labels:
                stats["discarded"] += 1
                stats["discard_reason"]["too_few_labels"] += 1
                if combo:
                    discarded_too_few_counter[combo] += 1
                continue

            if len(specifics) == 0:
                stats["discarded"] += 1
                stats["discard_reason"]["missing_specific"] += 1
                if combo:
                    discarded_only_general_counter[combo] += 1
                continue

            if len(generals) == 0:
                stats["discarded"] += 1
                stats["discard_reason"]["missing_general"] += 1
                if combo:
                    discarded_only_specific_counter[combo] += 1
                continue


            kept_data.append(item)
            stats["kept"] += 1

            kept_combo_counter[combo] += 1

            for sp in specifics:
                kept_specific_general_pattern_counter[(sp, tuple(generals))] += 1
                kept_specific_to_general_sets[sp][tuple(generals)] += 1

        except Exception:
            stats["discarded"] += 1
            stats["discard_reason"]["parse_error"] += 1
            continue


    print(f"\nFiltered data saved to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(kept_data, f_out, ensure_ascii=False, indent=2)


    print("\n" + "=" * 60)
    print(f"Gold-data filter report (min_labels={min_labels})")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Retained: {stats['kept']} ({(stats['kept'] / stats['total']) * 100:.2f}%)")
    print(f"Discarded: {stats['discarded']}")
    print("-" * 60)
    print(f"Too few labels (<{min_labels}): {stats['discard_reason']['too_few_labels']}")
    print(f"Missing specific article: {stats['discard_reason']['missing_specific']}")
    print(f"Missing general article: {stats['discard_reason']['missing_general']}")
    print(f"Parse errors: {stats['discard_reason']['parse_error']}")
    print("=" * 60)

    def print_top_counter(title, counter_obj, k=top_k):
        print(f"\n{title} (top {k})")
        print("-" * 60)
        if not counter_obj:
            print("No data")
            return
        for idx, (key, cnt) in enumerate(counter_obj.most_common(k), 1):
            print(f"{idx:02d}. {key} -> {cnt}")


    print_top_counter("Frequent complete article combinations", kept_combo_counter)
    print_top_counter("Frequent discarded specific-only combinations", discarded_only_specific_counter)
    print_top_counter("Frequent discarded general-only combinations", discarded_only_general_counter)
    print_top_counter("Frequent combinations with too few labels", discarded_too_few_counter)
    print_top_counter("Frequent specific/general co-occurrences", kept_specific_general_pattern_counter)


    print("\n" + "=" * 60)
    print("General-article combinations by frequent specific article")
    print("=" * 60)
    top_specifics = Counter()
    for (sp, gens), cnt in kept_specific_general_pattern_counter.items():
        top_specifics[sp] += cnt

    for sp, total_cnt in top_specifics.most_common(min(top_k, len(top_specifics))):
        print(f"\nSpecific article {sp} (occurrences: {total_cnt})")
        for gens, cnt in kept_specific_to_general_sets[sp].most_common(5):
            print(f"   General {gens} -> {cnt}")


    if stats_output_path:
        result = {
            "summary": stats,
            "top_kept_combinations": [
                {"combo": list(combo), "count": cnt}
                for combo, cnt in kept_combo_counter.most_common(top_k)
            ],
            "top_discarded_only_specific": [
                {"combo": list(combo), "count": cnt}
                for combo, cnt in discarded_only_specific_counter.most_common(top_k)
            ],
            "top_discarded_only_general": [
                {"combo": list(combo), "count": cnt}
                for combo, cnt in discarded_only_general_counter.most_common(top_k)
            ],
            "top_discarded_too_few": [
                {"combo": list(combo), "count": cnt}
                for combo, cnt in discarded_too_few_counter.most_common(top_k)
            ],
            "top_kept_specific_general_patterns": [
                {"specific": sp, "generals": list(gens), "count": cnt}
                for (sp, gens), cnt in kept_specific_general_pattern_counter.most_common(top_k)
            ]
        }

        print(f"\nStatistics saved to: {stats_output_path}")
        with open(stats_output_path, 'w', encoding='utf-8') as f_stat:
            json.dump(result, f_stat, ensure_ascii=False, indent=2)

    print("\nAnalysis complete.")
    return {
        "stats": stats,
        "kept_data": kept_data,
        "kept_combo_counter": kept_combo_counter,
        "discarded_only_specific_counter": discarded_only_specific_counter,
        "kept_specific_general_pattern_counter": kept_specific_general_pattern_counter
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter LeCaRD-style data into a gold subset with general/specific article constraints."
    )
    parser.add_argument("--input_path", required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output_path", required=True, help="Filtered output JSON path")
    parser.add_argument(
        "--stats_output_path",
        default=None,
        help="Optional JSON path for saving analysis statistics"
    )
    parser.add_argument(
        "--min_labels",
        type=int,
        default=3,
        help="Minimum number of article labels required to keep a sample"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many top combinations/patterns to print and save"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter_and_analyze(
        input_path=args.input_path,
        output_path=args.output_path,
        stats_output_path=args.stats_output_path,
        min_labels=args.min_labels,
        top_k=args.top_k,
    )
