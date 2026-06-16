"""Otimização Bayesiana de hiperparâmetros do multi-GRU (Tabela 5 do artigo) com Optuna.

Objetivo: testar se uma busca própria de hiperparâmetros **reabilita a vantagem do
multi-GRU** sobre o single-GRU (que na Fase 3b superou o multi sob a Tabela 9).

Faixas (Tabela 5 do artigo):
    hscale [2,10] (int) | lr [0.001,0.01] | beta1 [0.9,0.9999] | beta2 [0.9,0.9999]

Para ser tratável em CPU: subamostra o treino, usa poucas épocas por trial e poda
(MedianPruner) os trials ruins. Após a busca, treina o melhor config até convergir.

Uso:
    python -m src.models.optimize --trials 20 --trial-epochs 20 --subsample 0.25
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import preprocess
from .multi_gru import MultiGRU

SCENARIOS = ["straight_constant_speed", "straight", "jam", "bend"]


def _subset(X, M, y, frac, seed):
    if frac >= 1.0:
        return X, M, y
    rng = np.random.default_rng(seed)
    n = int(len(y) * frac)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], M[idx], y[idx]


def _loader(X, M, y, bs, shuffle):
    tds = TensorDataset(torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y))
    return DataLoader(tds, batch_size=bs, shuffle=shuffle)


def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, M, y in loader:
            pred = model(X.to(device), M.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def make_objective(Xtr, Mtr, ytr, Xva, Mva, yva, device, trial_epochs, batch_size):
    import optuna

    def objective(trial):
        hscale = trial.suggest_int("hscale", 2, 10)
        lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
        beta1 = trial.suggest_float("beta1", 0.9, 0.9999)
        beta2 = trial.suggest_float("beta2", 0.9, 0.9999)

        torch.manual_seed(42)
        model = MultiGRU(hscale=hscale).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
        crit = nn.CrossEntropyLoss()
        tr = _loader(Xtr, Mtr, ytr, batch_size, True)
        va = _loader(Xva, Mva, yva, batch_size, False)

        for ep in range(trial_epochs):
            model.train()
            for X, M, y in tr:
                opt.zero_grad()
                crit(model(X.to(device), M.to(device)), y.to(device)).backward()
                opt.step()
            acc = _accuracy(model, va, device)
            trial.report(acc, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return acc

    return objective


def run(trials=20, trial_epochs=20, subsample=0.25, batch_size=128,
        scenarios=SCENARIOS, seed=42, final_epochs=120):
    import optuna
    from sklearn.model_selection import train_test_split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Pré-processando {scenarios} ...")
    ds = preprocess.build_ai_dataset(scenarios=scenarios, seed=seed)

    # subamostra o treino e separa validação interna (test fica intocado)
    Xs, Ms, ys = _subset(ds.X_train, ds.M_train, ds.y_train, subsample, seed)
    Xtr, Xva, Mtr, Mva, ytr, yva = train_test_split(
        Xs, Ms, ys, test_size=0.25, random_state=seed, stratify=ys)
    print(f"     busca em: train={len(ytr)} val={len(yva)} (subsample={subsample})")

    print(f"[2/3] Optuna: {trials} trials × {trial_epochs} épocas ...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    obj = make_objective(Xtr, Mtr, ytr, Xva, Mva, yva, device, trial_epochs, batch_size)
    study.optimize(obj, n_trials=trials, show_progress_bar=False)

    print("\n=== MELHOR CONFIG ===")
    print(f"val_accuracy: {study.best_value*100:.4f}%")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\n[3/3] Treinando o melhor config no dataset completo ({final_epochs} épocas) ...")
    from . import train
    bp = study.best_params
    hp = {"hscale": bp["hscale"], "lr": bp["lr"], "beta1": bp["beta1"], "beta2": bp["beta2"]}
    res = train.run(scenarios=scenarios, epochs=final_epochs, batch_size=batch_size,
                    model_type="multi", hp=hp, save_as="multi_gru_bayes.pt", verbose=True)
    m = res["metrics"]
    print("\n=== RESULTADO multi-GRU (hiperparâmetros otimizados) ===")
    print(f"Accuracy: {m['accuracy']*100:.4f}%")
    for cls, d in m["per_class"].items():
        print(f"  {cls:7s} recall={d['recall']*100:6.2f}  f1={d['f1']:.4f}")
    return study, res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--trial-epochs", type=int, default=20)
    p.add_argument("--subsample", type=float, default=0.25)
    p.add_argument("--final-epochs", type=int, default=120)
    args = p.parse_args()
    run(trials=args.trials, trial_epochs=args.trial_epochs,
        subsample=args.subsample, final_epochs=args.final_epochs)
