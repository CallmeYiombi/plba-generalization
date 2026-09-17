"""Merge seed-sharded benchmark outputs into canonical multi-seed results."""
import argparse
import json

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

    # Top-up workers (e.g. --models Ligand-mean XGB-ESMonly) repeat a seed that
    # an earlier worker already ran, so report each seed once, and list every
    # model that actually has results, in the order the workers declare them.
    unique_seeds = list(dict.fromkeys(expected_seeds))
    lists = []
    for tag in worker_tags:
        with open(tagged_path("all_results_multiseed.json", tag)) as handle:
            lists.append(json.load(handle).get("models", []))
    # The most complete declaration is the newest MODEL_ORDER; use it first.
    declared = [m for lst in sorted(lists, key=len, reverse=True) for m in lst]
    models = [m for m in dict.fromkeys(declared) if m in all_results]
    models += [m for m in all_results if m not in models]

    combined = {
        "seeds": unique_seeds,
        "metrics": template.get("metrics", METRICS),
        "subsets": template.get("subsets", []),
        "models": models,
        "all_results": all_results,
        "evaluation_meta": evaluation_meta,
        "cold_start_meta": cold_start_meta,
        "failed_smiles_count": template.get("failed_smiles_count", {}),
    }
    return combined


def merge_tagged_csvs(filename, worker_tags, seeds, only_seed=None, key=None,
                      model_order=None):
    """Concatenate per-worker CSVs, optionally keeping one seed's workers only.

    Bootstrap intervals are computed on the last seed only, so every worker that
    ran that seed contributes rows (an original run plus any top-up run). Later
    workers win if the same key appears twice.
    """
    frames = []
    for tag, seed in zip(worker_tags, seeds):
        if only_seed is not None and seed != only_seed:
            continue
        path = tagged_path(filename, tag)
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    if key:
        merged = merged.drop_duplicates(key, keep="last")
    if model_order and "model" in merged.columns:
        rank = {m: i for i, m in enumerate(model_order)}
        merged = (merged.assign(_r=merged["model"].map(rank))
                  .sort_values("_r", kind="stable").drop(columns="_r"))
    return merged.reset_index(drop=True)


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

    bootstrap = merge_tagged_csvs(
        "results_bootstrap_ci.csv", args.worker_tags, args.seeds,
        only_seed=max(args.seeds), key=["model", "split", "subset"],
        model_order=combined["models"])
    if bootstrap is not None:
        bootstrap.to_csv(OUTPUT_DIR / "results_bootstrap_ci.csv", index=False)
    coverage = merge_tagged_csvs(
        "ligand_mean_coverage.csv", args.worker_tags, args.seeds)
    if coverage is not None:
        coverage.to_csv(OUTPUT_DIR / "ligand_mean_coverage.csv", index=False)

    print(f"Merged {len(args.worker_tags)} workers: seeds={args.seeds}")
    print(f"Saved: {OUTPUT_DIR / 'results_multiseed_summary.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'all_results_multiseed.json'}")


if __name__ == "__main__":
    main()
