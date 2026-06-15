"""Pré-processamento da Camada de IA (multi-GRU).

Reproduz a Seção 4.5 do artigo:
  1. Normalização min-max **por Message ID** (cada ID tem grandezas físicas diferentes).
  2. Janelamento em sequências (len=91, step=30).
  3. Split treino/teste 80/20.

O payload já vem deserializado nos CSVs como signal1..signal6 (Seção 4.5.1 já feita).

Projetado para uso em notebook:
    from src.data import load, preprocess
    ds = preprocess.build_ai_dataset(scenarios=["bend"])
    ds["X_train"].shape  # (n_seq, 91, 6)

Decisões de reprodução (o artigo não detalha tudo — ajustáveis):
  - Janelas NÃO cruzam fronteira de arquivo (cada CSV é um fluxo temporal contínuo).
  - Rótulo da janela = `any_attack`: se a janela contém >=1 pacote de ataque,
    recebe o label do ataque; senão, Normal. (alternativa: 'center').
  - min-max ajustado no conjunto inteiro fornecido (como no artigo). Para rigor
    estatístico, pode-se ajustar só no treino — ver `fit_minmax_per_id`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import load
from .load import SIGNAL_COLS, ID_TO_IDX, LABEL_NORMAL

SEQ_LEN = 91
STEP = 30


# --------------------------------------------------------------------------- #
# Normalização min-max por Message ID
# --------------------------------------------------------------------------- #
def fit_minmax_per_id(df: pd.DataFrame, signal_cols=SIGNAL_COLS) -> dict:
    """Calcula (min, max) de cada signal, por Message ID. Retorna dict[mid] -> (min, max)."""
    params = {}
    for mid, g in df.groupby("message_id"):
        vals = g[signal_cols].to_numpy(dtype=float)
        params[mid] = (vals.min(axis=0), vals.max(axis=0))
    return params


def apply_minmax_per_id(df: pd.DataFrame, params: dict, signal_cols=SIGNAL_COLS) -> pd.DataFrame:
    """Aplica a normalização min-max por Message ID. Sinais inválidos (sempre 0) permanecem 0."""
    out = df.copy()
    arr = out[signal_cols].to_numpy(dtype=float)
    mids = out["message_id"].to_numpy()
    for mid, (mn, mx) in params.items():
        mask = mids == mid
        rng = np.where(mx - mn == 0, 1.0, mx - mn)  # evita divisão por zero
        arr[mask] = (arr[mask] - mn) / rng
    out[signal_cols] = arr
    return out


# --------------------------------------------------------------------------- #
# Janelamento
# --------------------------------------------------------------------------- #
def make_windows(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    step: int = STEP,
    label_strategy: str = "any_attack",
    attack_label: int | None = None,
):
    """Gera janelas deslizantes sobre o fluxo temporal de pacotes.

    Retorna:
      X   : (n_seq, seq_len, 6)  sinais normalizados
      M   : (n_seq, seq_len)     índice do Message ID por pacote (p/ rotear no multi-GRU)
      y   : (n_seq,)             label da janela
    """
    df = df.sort_values("time")
    sig = df[SIGNAL_COLS].to_numpy(dtype=float)
    mid_idx = df["message_id"].map(ID_TO_IDX).to_numpy()
    labels = df["label"].to_numpy()
    n = len(df)

    X, M, Y = [], [], []
    for start in range(0, n - seq_len + 1, step):
        sl = slice(start, start + seq_len)
        win_labels = labels[sl]
        if label_strategy == "any_attack":
            if attack_label is None:
                raise ValueError("attack_label é obrigatório p/ label_strategy='any_attack'")
            y = attack_label if (win_labels == attack_label).any() else LABEL_NORMAL
        elif label_strategy == "center":
            y = int(win_labels[seq_len // 2])
        else:
            raise ValueError(f"label_strategy desconhecida: {label_strategy}")
        X.append(sig[sl])
        M.append(mid_idx[sl])
        Y.append(y)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(M, dtype=np.int64),
        np.asarray(Y, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Builder de alto nível
# --------------------------------------------------------------------------- #
@dataclass
class AIDataset:
    X_train: np.ndarray
    M_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    M_test: np.ndarray
    y_test: np.ndarray
    minmax: dict

    def __getitem__(self, k):  # permite ds["X_train"]
        return getattr(self, k)

    def summary(self) -> str:
        import collections
        tr = collections.Counter(self.y_train.tolist())
        te = collections.Counter(self.y_test.tolist())
        return (
            f"train: {self.X_train.shape} labels={dict(tr)}\n"
            f"test : {self.X_test.shape} labels={dict(te)}"
        )


def build_ai_dataset(
    scenarios=load.SCENARIOS,
    conditions=("n", "t", "r"),
    seq_len: int = SEQ_LEN,
    step: int = STEP,
    test_size: float = 0.2,
    seed: int = 42,
) -> AIDataset:
    """Pipeline completo: carrega -> normaliza -> janela -> split 80/20.

    - Normalização ajustada no conjunto inteiro (como no artigo).
    - Janelas geradas por (cenário, condição) para não cruzar arquivos.
    - Cada condição usa seu attack_label (_n normal, _t tamper, _r replay).
    """
    # 1) carrega tudo o que foi pedido
    frames = []
    for sc in scenarios:
        for cond in conditions:
            df = load.load_ai_csv(sc, cond)
            df["_scenario"] = sc
            df["_condition"] = cond
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # 2) min-max por Message ID (ajuste global)
    minmax = fit_minmax_per_id(full)

    # 3) janelas por (cenário, condição)
    Xs, Ms, Ys = [], [], []
    for (sc, cond), g in full.groupby(["_scenario", "_condition"]):
        g = apply_minmax_per_id(g, minmax)
        attack_label = load.CONDITION_ATTACK_LABEL.get(cond)  # None p/ 'n'
        strategy = "center" if attack_label is None else "any_attack"
        X, M, y = make_windows(g, seq_len, step, strategy, attack_label)
        Xs.append(X); Ms.append(M); Ys.append(y)

    X = np.concatenate(Xs); M = np.concatenate(Ms); y = np.concatenate(Ys)

    # 4) split 80/20 estratificado
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    return AIDataset(
        X_train=X[tr], M_train=M[tr], y_train=y[tr],
        X_test=X[te], M_test=M[te], y_test=y[te],
        minmax=minmax,
    )


if __name__ == "__main__":
    ds = build_ai_dataset(scenarios=["bend"])
    print(ds.summary())
