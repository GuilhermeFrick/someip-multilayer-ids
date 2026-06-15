"""Carga dos CSVs do dataset SOME-IP-IDS.

Reproduz o pré-passo de extração: lê os CSVs brutos e constrói o `message_id`
(service_id || method_id), que é a chave de roteamento tanto da Camada 1 (regras)
quanto da Camada 2 (multi-GRU).

Ver docs/descricao-dataset.md para a semântica das colunas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Raiz do projeto = dois níveis acima deste arquivo (src/data/load.py)
ROOT = Path(__file__).resolve().parents[2]
AI_DIR = ROOT / "data" / "raw" / "ai_detection"
RULE_DIR = ROOT / "data" / "raw" / "rule_detection"

SCENARIOS = ("bend", "jam", "straight", "straight_constant_speed")
CONDITIONS = {"n": "normal", "t": "tamper", "r": "replay"}

# Sinais por Message ID (event) — ver docs/descricao-dataset.md
SIGNALS_PER_ID = {
    "0x14720011": 2,
    "0x14720012": 2,
    "0x27590010": 6,
    "0x36120009": 3,
}

# Colunas de payload deserializado presentes nos CSVs
SIGNAL_COLS = [f"signal{i}" for i in range(1, 7)]

# Mapeamento Message ID -> índice (ordem fixa p/ rotear no multi-GRU)
EVENT_IDS = list(SIGNALS_PER_ID.keys())
ID_TO_IDX = {mid: i for i, mid in enumerate(EVENT_IDS)}

# Labels (parte de IA)
LABEL_NORMAL = 1
LABEL_TAMPER = 0
LABEL_REPLAY = 2
# Label de ataque associado a cada condição de arquivo (_t -> tamper, _r -> replay)
CONDITION_ATTACK_LABEL = {"t": LABEL_TAMPER, "r": LABEL_REPLAY}


def _add_message_id(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a coluna `message_id` = service_id (4 hex) + method_id (4 hex)."""
    sid = df["service_id"].astype(str).str.replace("0x", "", regex=False).str.zfill(4)
    mid = df["method_id"].astype(str).str.replace("0x", "", regex=False).str.zfill(4)
    df["message_id"] = "0x" + (sid + mid).str.upper()
    return df


def load_ai_csv(scenario: str, condition: str) -> pd.DataFrame:
    """Carrega um CSV da parte de IA. condition ∈ {'n','t','r'}."""
    path = AI_DIR / f"data_{scenario}_4_08_someip_{condition}.csv"
    df = pd.read_csv(path, index_col=0)
    return _add_message_id(df)


def load_ai_all() -> pd.DataFrame:
    """Concatena todos os 12 CSVs de IA, anotando cenário e condição."""
    frames = []
    for sc in SCENARIOS:
        for cond in CONDITIONS:
            df = load_ai_csv(sc, cond)
            df["scenario"] = sc
            df["condition"] = CONDITIONS[cond]
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_rule_someip() -> pd.DataFrame:
    df = pd.read_csv(
        RULE_DIR / "SOMEIPHeader_rule_someip.csv", index_col=0, low_memory=False
    )
    # coluna fantasma da vírgula final no header (quase toda NaN) — não é label
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return _add_message_id(df)


def load_rule_sd() -> pd.DataFrame:
    df = pd.read_csv(
        RULE_DIR / "SOMEIPHeader_rule_SD.csv", index_col=0, low_memory=False
    )
    return _add_message_id(df)


if __name__ == "__main__":
    # Sanity check rápido
    df = load_ai_csv("bend", "t")
    print("Colunas:", list(df.columns))
    print("Message IDs:", sorted(df["message_id"].unique()))
    print("Labels:", df["label"].value_counts().to_dict())
    print("Shape:", df.shape)
