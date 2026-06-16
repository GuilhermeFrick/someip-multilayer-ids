"""Treino e avaliação do multi-GRU (Seção 5.4 do artigo).

Otimizador Adam + cross-entropy. Hiperparâmetros default = Tabela 9 (multi-GRU).

Uso (notebook):
    from src.models import train
    res = train.run(scenarios=["bend"], epochs=30)
    print(res["metrics"])

Uso (CLI):
    python -m src.models.train
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import preprocess
from ..data.load import LABEL_TAMPER, LABEL_NORMAL, LABEL_REPLAY
from .multi_gru import MultiGRU, count_parameters
from .single_gru import SingleGRU

ROOT = Path(__file__).resolve().parents[2]

# Hiperparâmetros — Tabela 9 do artigo
HP_MULTI = dict(hscale=5, lr=0.0089630704, beta1=0.933792409392, beta2=0.952802490181)
HP_SINGLE = dict(hscale=31, lr=0.0043259137, beta1=0.939844012507, beta2=0.943045819607)
HP_DEFAULTS = {"multi": HP_MULTI, "single": HP_SINGLE}


def build_model(model_type: str, hscale: int):
    if model_type == "multi":
        return MultiGRU(hscale=hscale)
    if model_type == "single":
        return SingleGRU(hscale=hscale)
    raise ValueError(f"model_type inválido: {model_type!r} (use 'multi' ou 'single')")

# Mapeia o label original {0,1,2} -> índice de classe contíguo {0,1,2} p/ cross-entropy.
# Mantemos a mesma numeração (já é 0..2), mas deixamos explícito p/ legibilidade.
CLASSES = [LABEL_TAMPER, LABEL_NORMAL, LABEL_REPLAY]  # 0,1,2
CLASS_NAMES = {LABEL_TAMPER: "Tamper", LABEL_NORMAL: "Normal", LABEL_REPLAY: "Replay"}


def _loaders(ds, batch_size: int):
    def make(X, M, y, shuffle):
        tds = TensorDataset(
            torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)
        )
        return DataLoader(tds, batch_size=batch_size, shuffle=shuffle)

    return (
        make(ds.X_train, ds.M_train, ds.y_train, True),
        make(ds.X_test, ds.M_test, ds.y_test, False),
    )


def evaluate(model, loader, device) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
    )

    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for X, M, y in loader:
            logits = model(X.to(device), M.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            CLASS_NAMES[c]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def run(scenarios=("bend",), epochs: int = 30, batch_size: int = 128,
        model_type: str = "multi", hp: dict | None = None, seed: int = 42,
        save_as: str | None = None, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    hp = {**HP_DEFAULTS[model_type], **(hp or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"[1/3] Pré-processando cenários={list(scenarios)} ...")
    ds = preprocess.build_ai_dataset(scenarios=list(scenarios), seed=seed)
    if verbose:
        print("     ", ds.summary().replace("\n", "\n      "))

    train_loader, test_loader = _loaders(ds, batch_size)
    model = build_model(model_type, hp["hscale"]).to(device)
    if verbose:
        print(f"[2/3] Treinando {model_type}-GRU ({count_parameters(model)} parâmetros) "
              f"em {device} por {epochs} épocas ...")

    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], betas=(hp["beta1"], hp["beta2"]))
    crit = nn.CrossEntropyLoss()

    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for X, M, y in train_loader:
            opt.zero_grad()
            loss = crit(model(X.to(device), M.to(device)), y.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        ep_loss = tot / len(train_loader.dataset)
        losses.append(ep_loss)
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"     época {ep:3d}  loss={ep_loss:.5f}")

    if verbose:
        print("[3/3] Avaliando no conjunto de teste ...")
    metrics = evaluate(model, test_loader, device)

    if save_as:
        out = ROOT / "models" / save_as
        torch.save({"state_dict": model.state_dict(), "hp": hp,
                    "metrics": metrics}, out)
        if verbose:
            print("     modelo salvo em", out)

    return {"model": model, "metrics": metrics, "losses": losses, "hp": hp}


if __name__ == "__main__":
    res = run(scenarios=["bend"], epochs=30, save_as="multi_gru_bend.pt")
    m = res["metrics"]
    print("\n=== RESULTADO ===")
    print(f"Accuracy: {m['accuracy']*100:.4f}%")
    for cls, d in m["per_class"].items():
        print(f"  {cls:7s} precision={d['precision']*100:6.2f}  "
              f"recall={d['recall']*100:6.2f}  f1={d['f1']:.4f}")
    print("Matriz de confusão (linhas=verdadeiro, ordem Tamper/Normal/Replay):")
    for row in m["confusion_matrix"]:
        print("  ", row)
