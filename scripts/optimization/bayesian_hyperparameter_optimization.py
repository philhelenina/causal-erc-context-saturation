#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_hyperparameter_optimization.py
Runs Optuna search for best hyperparameters using train_turn_classifier.py
"""

import optuna
import json
import os
from pathlib import Path
from train_turn_classifier import train_and_evaluate_once

HOME = Path(os.environ.get("SENTICCRYSTAL_ROOT", Path.cwd())).resolve()
RESULT_DIR = HOME / "results" / "bayesian_optimization"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: WandB integration
USE_WANDB = False  # Set to True if you want WandB logging
if USE_WANDB:
    try:
        import wandb
    except ImportError:
        print("[WARNING] wandb not installed, disabling WandB logging")
        USE_WANDB = False


def objective(trial, task, embedding, layer, pool, model):
    """
    Objective function for Optuna optimization
    
    Args:
        trial: Optuna trial object
        task: '4way' or '6way'
        embedding: encoder name
        layer: layer config
        pool: pooling method
        model: 'mlp' or 'lstm'
    
    Returns:
        f1_weighted: Objective to maximize
    """
    # Search space (updated to use modern Optuna API)
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 192, 256, 384]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.8),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "num_epochs": trial.suggest_int("num_epochs", 50, 150),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.01),
        "early_stopping_patience": 20,
    }
    
    print(f"\n[Trial {trial.number}]")
    print(f"  lr={params['learning_rate']:.6f}, hidden={params['hidden_size']}, "
          f"dropout={params['dropout_rate']:.3f}, bs={params['batch_size']}")
    
    # Run training
    try:
        metrics = train_and_evaluate_once(task, embedding, layer, pool, model, params)
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        return 0.0
    
    f1w = metrics.get("f1_weighted", 0.0)
    f1m = metrics.get("f1_macro", 0.0)
    acc = metrics.get("acc", 0.0)
    
    print(f"  → F1w={f1w:.4f}, F1m={f1m:.4f}, Acc={acc:.4f}")
    
    # Log to WandB if enabled
    if USE_WANDB:
        wandb.log({
            "trial": trial.number,
            "f1_weighted": f1w,
            "f1_macro": f1m,
            "acc": acc,
            **params
        })
    
    return f1w  # Maximize weighted F1


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for utterance-level baseline"
    )
    ap.add_argument("--task", choices=["4way", "6way"], required=True,
                    help="Classification task")
    ap.add_argument("--embedding", default="sentence-roberta-hier",
                    help="Encoder name (default: sentence-roberta-hier)")
    ap.add_argument("--layer", required=True,
                    help="Layer config (e.g., avg_last4)")
    ap.add_argument("--pool", required=True,
                    help="Pooling method (e.g., mean)")
    ap.add_argument("--model", choices=["mlp", "lstm"], default="mlp",
                    help="Classifier type (default: mlp)")
    ap.add_argument("--n_trials", type=int, default=20,
                    help="Number of optimization trials (default: 20)")
    ap.add_argument("--study_name", required=True,
                    help="Study name for Optuna")
    ap.add_argument("--wandb_project", default="senticcrystal-hyperopt",
                    help="WandB project name (only if USE_WANDB=True)")
    ap.add_argument("--gpu", default="0",
                    help="GPU device ID (default: 0)")
    args = ap.parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    print("="*80)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Embedding: {args.embedding}")
    print(f"Layer: {args.layer}")
    print(f"Pool: {args.pool}")
    print(f"Model: {args.model}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Study name: {args.study_name}")
    print(f"GPU: {args.gpu}")
    print("="*80)
    print()
    
    # Initialize WandB if enabled
    if USE_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=args.study_name,
            config=vars(args)
        )
        print("[INFO] WandB logging enabled")
    else:
        print("[INFO] WandB logging disabled")
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=optuna.samplers.TPESampler(seed=42)  # For reproducibility
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.task, args.embedding, args.layer, args.pool, args.model),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"Best F1 weighted: {best_value:.4f}")
    print()
    print("Best hyperparameters:")
    print(json.dumps(best_params, indent=2))
    print()
    
    # Save results
    out_file = RESULT_DIR / f"bayesopt_{args.task}_{args.embedding}_{args.layer}_{args.pool}_{args.model}.json"
    
    results = {
        "task": args.task,
        "embedding": args.embedding,
        "layer": args.layer,
        "pool": args.pool,
        "model": args.model,
        "n_trials": args.n_trials,
        "best_f1_weighted": float(best_value),
        "best_params": best_params,
        "study_name": args.study_name
    }
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Saved results → {out_file}")
    
    # Save Optuna study
    study_file = RESULT_DIR / f"{args.study_name}_study.pkl"
    import pickle
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    print(f"[OK] Saved study → {study_file}")
    
    print()
    print("="*80)


if __name__ == "__main__":
    main()