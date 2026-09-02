"""Merge seed-sharded benchmark outputs into canonical multi-seed results."""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from config import OUTPUT_DIR


METRICS = ["PCC", "SRCC", "RMSE", "R2", "CI"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-tags", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser.parse_args()


def tagged_path(filename, tag):
    path = OUTPUT_DIR / filename
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def merge_worker_jsons(worker_tags, expected_seeds):
    all_results = {}
    evaluation_meta = {}
    cold_start_meta = {}
    template = None

    for tag in worker_tags:
        path = tagged_path("all_results_multiseed.json", tag)
        if not path.exists():
            raise FileNotFoundError(f"Missing worker result: {path}")
        with open(path) as handle:
            data = json.load(handle)
        template = template or data

        for model, split_dict in data["all_results"].items():
            for split, seed_dict in split_dict.items():
                target = all_results.setdefault(model, {}).setdefault(split, {})
                for seed, subset_metrics in seed_dict.items():
                    if seed in target:
                        raise ValueError(
                            f"Duplicate result for model={model}, split={split}, seed={seed}"
                        )
                    target[seed] = subset_metrics

        evaluation_meta.update(data.get("evaluation_meta", {}))
        cold_start_meta.update(data.get("cold_start_meta", {}))

    expected = {str(seed) for seed in expected_seeds}
    for model, split_dict in all_results.items():
        for split, seed_dict in split_dict.items():
            observed = set(seed_dict)
            if observed != expected:
                raise RuntimeError(
                    f"Incomplete results for {model}/{split}: "
                    f"expected {sorted(expected)}, observed {sorted(observed)}"
                )

    combined = {
        "seeds": expected_seeds,
        "metrics": template.get("metrics", METRICS),
        "subsets": template.get("subsets", []),
        "models": template.get("models", []),
        "all_results": all_results,
        "evaluation_meta": evaluation_meta,
        "cold_start_meta": cold_start_meta,
        "failed_smiles_count": template.get("failed_smiles_count", {}),
    }
    return combined


def summarize(all_results):
    rows = []
    for model, split_dict in all_results.items():
        for split, seed_dict in split_dict.items():
            subset_metrics = {}
            for results in seed_dict.values():
                for subset_name, metrics in results.items():
                    for metric, value in metrics.items():
                        (subset_metrics.setdefault(subset_name, {})
                         .setdefault(metric, [])).append(value)
            for subset_name, metric_dict in subset_metrics.items():
                row = {
                    "model": model,
                    "split": split,
                    "subset": subset_name,
                    "n_seeds": len(metric_dict.get("PCC", [])),
                }
                for metric, values in metric_dict.items():
                    row[f"{metric}_mean"] = round(float(np.mean(values)), 4)
                    row[f"{metric}_std"] = round(float(np.std(values)), 4)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if len(args.worker_tags) != len(args.seeds):
        raise ValueError("--worker-tags and --seeds must have the same length")

    combined = merge_worker_jsons(args.worker_tags, args.seeds)
    summary = summarize(combined["all_results"])
    summary.to_csv(OUTPUT_DIR / "results_multiseed_summary.csv", index=False)

    with open(OUTPUT_DIR / "all_results_multiseed.json", "w") as handle:
        json.dump(combined, handle, indent=2)

    last_seed_idx = int(np.argmax(args.seeds))
    last_tag = args.worker_tags[last_seed_idx]
    bootstrap_source = tagged_path("results_bootstrap_ci.csv", last_tag)
    if bootstrap_source.exists():
        shutil.copyfile(bootstrap_source, OUTPUT_DIR / "results_bootstrap_ci.csv")

    print(f"Merged {len(args.worker_tags)} workers: seeds={args.seeds}")
    print(f"Saved: {OUTPUT_DIR / 'results_multiseed_summary.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'all_results_multiseed.json'}")


if __name__ == "__main__":
    main()
