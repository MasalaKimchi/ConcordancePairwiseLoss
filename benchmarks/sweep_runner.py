#!/usr/bin/env python3
"""
Hyperparameter sweep runner for ConcordancePairwiseLoss experiments.

This script does NOT modify benchmark_framework.py. It orchestrates multiple
runs of benchmark_framework.py with different hyperparameter combinations and
aggregates results. It is designed to be simple and dependency-free.

Usage examples (PowerShell/Windows):
  conda activate concordance-pairwise-loss
  python benchmarks/sweep_runner.py --dataset gbsg2 --num-features 9 --sweep cphl --runs-per-trial 3 --epochs 30

  # CPL only (with and without IPCW)
  python benchmarks/sweep_runner.py --dataset gbsg2 --num-features 9 --sweep cpl --runs-per-trial 3 --epochs 30

Output: results are placed under benchmarks/results/sweeps/<dataset>/<sweep_id>/
and a summary JSON/CSV is produced aggregating Uno C across trials.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _glob_latest_json(directory: Path) -> Path:
    candidates = sorted(directory.glob("*_comprehensive.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _run_once(cmd: List[str]) -> int:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return proc.returncode


def _read_results(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)


def build_cpl_trials(args) -> List[Dict[str, Any]]:
    temps = [0.5, 1.0, 2.0]
    lrs = [5e-3, 1e-2, 5e-2]
    wds = [0.0, 1e-4, 1e-3]
    methods = ["pairwise", "pairwise_ipcw"]
    trials = []
    for method, t, lr, wd in itertools.product(methods, temps, lrs, wds):
        trials.append({
            "loss_types": [method],
            "temperature": t,
            "lr": lr,
            "weight_decay": wd,
        })
    return trials


def build_cphl_trials(args) -> List[Dict[str, Any]]:
    rels = [0.5, 0.75, 1.0, 1.5]
    temps = [0.5, 1.0, 2.0]
    hetero = [False, True]
    kinds = ["exp", "none"]
    lrs = [5e-3, 1e-2, 5e-2]
    wds = [0.0, 1e-4, 1e-3]
    trials = []
    for kind, r, t, h, lr, wd in itertools.product(kinds, rels, temps, hetero, lrs, wds):
        trials.append({
            "loss_types": [f"cphl_{kind}"],
            "rel_factor": r,
            "temperature": t,
            "hetero_tau": h,
            "lr": lr,
            "weight_decay": wd,
        })
    return trials


def build_hybrid_trials(args) -> List[Dict[str, Any]]:
    rels = [0.5, 0.75, 1.0]
    temps = [0.5, 1.0, 2.0]
    hetero = [False, True]
    kinds = ["exp", "none"]
    use_uw = [True, False]  # True: hybrid_kind; False: hybrid_kind_simple
    lrs = [5e-3, 1e-2, 5e-2]
    wds = [0.0, 1e-4, 1e-3]
    trials = []
    for kind, uw, r, t, h, lr, wd in itertools.product(kinds, use_uw, rels, temps, hetero, lrs, wds):
        loss_key = f"hybrid_{kind}" if uw else f"hybrid_{kind}_simple"
        trials.append({
            "loss_types": [loss_key],
            "rel_factor": r,
            "temperature": t,
            "hetero_tau": h,
            "uncertainty_weighting": uw,
            "lr": lr,
            "weight_decay": wd,
        })
    return trials


def hidden_dim_candidates(dataset: str) -> List[int]:
    small = {"gbsg2", "whas500"}
    large = {"support2"}
    if dataset.lower() in small:
        return [16, 32, 64]
    if dataset.lower() in large:
        return [128, 256]
    return [64, 128, 256]


def build_random_trials(args, family: str, n_trials: int) -> List[Dict[str, Any]]:
    import random
    random.seed(args.seed or 0)
    hiddens = hidden_dim_candidates(args.dataset)

    trials = []
    for _ in range(n_trials):
        lr = 10 ** random.uniform(-3.3, -1.3)  # ~[5e-4, 5e-2]
        t = random.choice([0.5, 1.0, 2.0])
        hd = random.choice(hiddens)
        if family == "cpl":
            method = random.choice(["pairwise", "pairwise_ipcw"])
            trials.append({"loss_types": [method], "temperature": t, "lr": lr, "hidden_dim": hd})
        elif family == "cphl":
            kind = random.choice(["exp", "none"])
            rel = random.choice([0.5, 0.75, 1.0, 1.5])
            hetero = random.choice([False, True])
            trials.append({
                "loss_types": [f"cphl_{kind}"], "rel_factor": rel, "temperature": t, "hetero_tau": hetero,
                "lr": lr, "hidden_dim": hd,
            })
        elif family == "hybrid":
            kind = random.choice(["exp", "none"])
            uw = random.choice([True, False])
            rel = random.choice([0.5, 0.75, 1.0])
            hetero = random.choice([False, True])
            loss_key = f"hybrid_{kind}" if uw else f"hybrid_{kind}_simple"
            trials.append({
                "loss_types": [loss_key], "rel_factor": rel, "temperature": t, "hetero_tau": hetero,
                "lr": lr, "hidden_dim": hd,
            })
    return trials


def summarize(directory: Path, dataset: str, summary_out: Path) -> None:
    """Scan all comprehensive JSONs in directory and summarize best by Uno C per method."""
    rows = []
    
    # Look for comprehensive JSONs first
    json_patterns = ["*_comprehensive.json", "*.json"]
    json_files = []
    for pattern in json_patterns:
        # Search recursively to include trial subdirectories
        json_files.extend(directory.rglob(pattern))
    
    if not json_files:
        print(f"Warning: No JSON files found in {directory}")
        out = {
            "dataset": dataset,
            "num_candidates": 0,
            "best_by_method": {},
            "error": "No JSON files found"
        }
        with open(summary_out, "w") as f:
            json.dump(out, f, indent=2)
        print("Summary saved:", summary_out)
        return
    
    for path in sorted(json_files, key=lambda p: p.stat().st_mtime):
        try:
            data = _read_results(path)
            perf = data.get("performance_summary", {})
            hypers = data.get("hyperparameters", {})
            for method_name, metrics in perf.items():
                if isinstance(metrics, dict):
                    rows.append({
                        "file": str(path),
                        "method": method_name,
                        "uno_cindex": metrics.get("uno_cindex"),
                        "harrell_cindex": metrics.get("harrell_cindex"),
                        "cumulative_auc": metrics.get("cumulative_auc"),
                        "incident_auc": metrics.get("incident_auc"),
                        "brier_score": metrics.get("brier_score"),
                        "hyperparameters": hypers,
                    })
        except Exception as e:
            print(f"Warning: Failed to process {path}: {e}")
            continue
    
    # Best by Uno per method
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        m = r["method"]
        if r["uno_cindex"] is None:
            continue
        if m not in best or r["uno_cindex"] > best[m]["uno_cindex"]:
            best[m] = r
    
    out = {
        "dataset": dataset,
        "num_candidates": len(rows),
        "best_by_method": best,
    }
    with open(summary_out, "w") as f:
        json.dump(out, f, indent=2)
    print("Summary saved:", summary_out)


def _method_display_name(loss_key: str) -> str:
    mapping = {
        'pairwise': 'CPL',
        'pairwise_ipcw': 'CPL (ipcw)',
        'normalized_combination': 'NLL+CPL',
        'normalized_combination_ipcw': 'NLL+CPL (ipcw)',
        'cphl_none': 'CPHL (none)',
        'cphl_exp': 'CPHL (exp)',
        'hybrid_none': 'Hybrid (none)',
        'hybrid_exp': 'Hybrid (exp)',
        'hybrid_exp_simple': 'Hybrid (exp, simple)',
    }
    return mapping.get(loss_key, loss_key)


def run_optuna_for_family(args, family: str, out_dir: Path, n_trials: int) -> None:
    try:
        import optuna
    except Exception:
        print("Optuna not available, falling back to random search for", family)
        trials = build_random_trials(args, family, n_trials)
        # Execute the random trials using existing loop
        for idx, t in enumerate(trials, start=1):
            trial_dir = out_dir / f"trial_{family}_{idx:04d}"
            _ensure_dir(trial_dir)
            cmd = [
                sys.executable, "benchmarks/benchmark_framework_ext.py",
                "--dataset", args.dataset,
                "--runs", str(args.runs_per_trial),
                "--epochs", str(args.epochs),
                "--num-features", str(args.num_features),
                "--output-dir", str(trial_dir),
            ]
            if t.get("lr") is not None:
                cmd += ["--lr", str(t["lr"])]
            if "hidden_dim" in t:
                cmd += ["--hidden-dim", str(t["hidden_dim"])]
            loss_types = t["loss_types"]
            cmd += ["--loss-types"] + loss_types
            if "temperature" in t:
                cmd += ["--temperature", str(t["temperature"])]
            if "rel_factor" in t:
                cmd += ["--rel-factor", str(t["rel_factor"])]
            if t.get("hetero_tau"):
                cmd += ["--hetero-tau"]
            rc = _run_once(cmd)
            if rc != 0:
                continue
        return

    def sample_and_run(trial: "optuna.trial.Trial") -> float:
        # Sample hyperparams per family
        hidden = trial.suggest_categorical("hidden_dim", hidden_dim_candidates(args.dataset))
        lr = trial.suggest_float("lr", 5e-4, 5e-2, log=True)
        temp = trial.suggest_categorical("temperature", [0.5, 1.0, 2.0])
        config = {"hidden_dim": hidden, "lr": lr, "temperature": temp}
        if family == "cpl":
            loss_key = trial.suggest_categorical("loss", ["pairwise", "pairwise_ipcw"])
        elif family == "cphl":
            kind = trial.suggest_categorical("kind", ["exp", "none"]).lower()
            loss_key = f"cphl_{kind}"
            config["rel_factor"] = trial.suggest_categorical("rel_factor", [0.5, 0.75, 1.0, 1.5])
            config["hetero_tau"] = trial.suggest_categorical("hetero_tau", [False, True])
        elif family == "hybrid":
            kind = trial.suggest_categorical("kind", ["exp", "none"]).lower()
            uw = trial.suggest_categorical("uncertainty_weighting", [True, False])
            loss_key = f"hybrid_{kind}" if uw else f"hybrid_{kind}_simple"
            config["rel_factor"] = trial.suggest_categorical("rel_factor", [0.5, 0.75, 1.0])
            config["hetero_tau"] = trial.suggest_categorical("hetero_tau", [False, True])
        else:
            loss_key = "pairwise"
        # Run trial
        trial_dir = out_dir / f"trial_{family}_optuna_{trial.number:04d}"
        _ensure_dir(trial_dir)
        cmd = [
            sys.executable, "benchmarks/benchmark_framework_ext.py",
            "--dataset", args.dataset,
            "--runs", str(args.runs_per_trial),
            "--epochs", str(args.epochs),
            "--num-features", str(args.num_features),
            "--output-dir", str(trial_dir),
            "--lr", str(config["lr"]),
            "--hidden-dim", str(config["hidden_dim"]),
            "--loss-types", loss_key,
            "--temperature", str(config["temperature"]),
        ]
        if "rel_factor" in config:
            cmd += ["--rel-factor", str(config["rel_factor"])]
        if config.get("hetero_tau"):
            cmd += ["--hetero-tau"]
        rc = _run_once(cmd)
        if rc != 0:
            raise optuna.TrialPruned(f"Runner failed with code {rc}")
        # Read latest comprehensive and return Uno C for the method
        latest = _glob_latest_json(trial_dir)
        if latest is None:
            print(f"Warning: No comprehensive JSON found in {trial_dir}")
            # Try to find any JSON file
            json_files = list(trial_dir.glob("*.json"))
            if json_files:
                latest = max(json_files, key=lambda p: p.stat().st_mtime)
                print(f"Using latest JSON: {latest}")
            else:
                raise optuna.TrialPruned("No result JSON found")
        
        data = _read_results(latest)
        perf = data.get("performance_summary", {})
        mname = _method_display_name(loss_key)
        uno = perf.get(mname, {}).get("uno_cindex")
        if uno is None:
            print(f"Warning: Uno C not found for {mname} in {latest}")
            print(f"Available methods: {list(perf.keys())}")
            # Try to find any method with uno_cindex
            for method, metrics in perf.items():
                if isinstance(metrics, dict) and "uno_cindex" in metrics:
                    uno = metrics["uno_cindex"]
                    print(f"Using {method} with Uno C: {uno}")
                    break
            if uno is None:
                raise optuna.TrialPruned("Uno not found")
        return float(uno)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(sample_and_run, n_trials=n_trials, show_progress_bar=False)
    print(f"Best {family} trial: value={study.best_value:.4f}, params={study.best_params}")


def main():
    parser = argparse.ArgumentParser(description="Sweep runner for CPL/CPHL/Hybrid")
    parser.add_argument("--dataset", required=True, help="Dataset key (matches DATA_LOADERS)")
    parser.add_argument("--num-features", type=int, required=True, help="Number of input features")
    parser.add_argument("--sweep", choices=["cpl", "cphl", "hybrid", "all"], default="all", help="Which family to sweep")
    parser.add_argument("--runs-per-trial", type=int, default=3, help="Num. of seeds per trial (passes --runs)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--output-root", type=str, default="benchmarks/results/sweeps", help="Root output directory")
    parser.add_argument("--seed", type=int, default=None, help="Base seed used (optional)")
    parser.add_argument("--num-trials", type=int, default=30, help="Optuna trials per family (default)")
    parser.add_argument("--strategy", choices=["optuna", "random"], default="optuna", help="Search strategy")
    args = parser.parse_args()

    sweep_id = _now_id()
    out_dir = Path(args.output_root) / args.dataset / sweep_id
    _ensure_dir(out_dir)
    print(f"Sweep output dir: {out_dir}")

    # Build trial grids
    if args.strategy == "optuna":
        families = [args.sweep] if args.sweep != "all" else ["cpl", "cphl", "hybrid"]
        for fam in families:
            run_optuna_for_family(args, fam, out_dir, args.num_trials)
    else:
        trials: List[Dict[str, Any]] = []
        per_family = 20
        if args.sweep in ("cpl", "all"):
            trials += build_random_trials(args, "cpl", per_family)
        if args.sweep in ("cphl", "all"):
            trials += build_random_trials(args, "cphl", per_family)
        if args.sweep in ("hybrid", "all"):
            trials += build_random_trials(args, "hybrid", per_family)
        print(f"Total trials: {len(trials)}")
        for idx, t in enumerate(trials, start=1):
            trial_dir = out_dir / f"trial_{idx:04d}"
            _ensure_dir(trial_dir)
            cmd = [
                sys.executable, "benchmarks/benchmark_framework_ext.py",
                "--dataset", args.dataset,
                "--runs", str(args.runs_per_trial),
                "--epochs", str(args.epochs),
                "--num-features", str(args.num_features),
                "--output-dir", str(trial_dir),
            ]
            lr = t.get("lr")
            if lr is not None:
                cmd += ["--lr", str(lr)]
            if "hidden_dim" in t:
                cmd += ["--hidden-dim", str(t["hidden_dim"])]
            loss_types = t["loss_types"]
            cmd += ["--loss-types"] + loss_types
            if "temperature" in t:
                cmd += ["--temperature", str(t["temperature"])]
            if "rel_factor" in t:
                cmd += ["--rel-factor", str(t["rel_factor"])]
            if t.get("hetero_tau"):
                cmd += ["--hetero-tau"]
            rc = _run_once(cmd)
            if rc != 0:
                print(f"Trial {idx} failed with code {rc}")
                continue

    # Summarize
    summary_path = out_dir / "summary_best_by_uno.json"
    summarize(out_dir, args.dataset, summary_path)
    print("Sweep finished.")


if __name__ == "__main__":
    main()


