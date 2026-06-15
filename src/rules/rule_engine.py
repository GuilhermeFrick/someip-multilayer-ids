"""Camada 1 — Detecção baseada em regras (Seção 4.4 do artigo).

Funciona como uma **whitelist**: cada Message ID tem um conjunto de regras (estáticas,
de estado e dinâmicas). Se qualquer regra falha, o pacote é anomalia.

Categorias de regra implementadas:
  - R_UNKNOWN_ID  (Fuzzy)    : (service_id, method_id) fora do conjunto conhecido.
  - R_STATIC      (Fuzzy)    : campo estático fora da whitelist do Message ID
                               (mac/porta/versões/client_id/someip_length/message_type).
  - R_RETURN_CODE (Abnormal) : return_code != 0x00 (erro -> processo de comunicação anormal).
  - R_STATE       (Abnormal) : missing request/response em RPC Request-Response, pareando
                               por (client_id, session_id) — independente de timestamp.
  - R_INTERVAL    (DoS)      : intervalo do pacote abaixo do ciclo esperado (best-effort).

------------------------------------------------------------------------------
LIMITAÇÃO DE AVALIAÇÃO (importante p/ a dissertação)
------------------------------------------------------------------------------
Os CSVs de regra do dataset **NÃO têm coluna de label** (normal/ataque). A "verdade"
no artigo é interna ao gerador. Portanto, NÃO há como calcular accuracy/recall por
pacote contra um ground-truth independente. O que fazemos:
  - construir a whitelist a partir do próprio tráfego usando **frequência** (os valores
    legítimos dominam ~45000x; os fuzzed aparecem ~2x), o que separa bem ataque de normal;
  - reportar a contagem de anomalias por tipo e comparar com a Tabela 7 do artigo
    (normal 55010, fuzzy 43867, dos 12188, abnormal 33509).
A whitelist por frequência é a forma reprodutível da "observação humana" descrita no artigo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Campos estáticos esperados constantes (ou multi-valor fixo) por Message ID.
SOMEIP_STATIC = [
    "mac_dst", "mac_src", "srcport", "dstport",
    "client_id", "protocol_version", "interface_version",
    "someip_length", "message_type",
]

# Campos estáticos do Service Discovery (whitelist própria).
SD_STATIC = [
    "mac_dst", "mac_src", "srcport", "dstport",
    "protocol_version", "interface_version", "message_type",
    "sd_entry1_type", "sd_entry1_sid", "sd_entry1_instance_id",
    "sd_entry1_major_version", "sd_entry1_ttl",
    "sd_option1_type", "sd_option1_ipv4", "sd_option1_port",
]

VALID_RETURN_CODES = ("0x00",)  # E_OK

# Tipos de mensagem SOME/IP relevantes p/ a máquina de estados
MSG_REQUEST = "0x00"   # REQUEST (espera RESPONSE)
MSG_RESPONSE = "0x80"  # RESPONSE


@dataclass
class RuleEngine:
    """Motor de regras com whitelist aprendida por frequência.

    Parâmetros:
      static_fields : campos estáticos a whitelistar por Message ID.
      min_freq      : fração mínima (por Message ID) p/ um valor entrar na whitelist.
      min_mid_count : contagem mínima p/ um (service_id, method_id) ser Message ID legítimo.
      interval_factor: fração do ciclo esperado abaixo da qual o intervalo é DoS.
      check_interval: liga/desliga a regra de DoS (best-effort; ver limitação).
      check_state   : liga/desliga a máquina de estados request↔response (R_STATE).
    """

    static_fields: list = field(default_factory=lambda: list(SOMEIP_STATIC))
    min_freq: float = 0.005
    min_mid_count: int = 100
    interval_factor: float = 0.5
    check_state: bool = True
    # DoS via intervalo é DESLIGADO por padrão: os timestamps do dataset (ms, com muitas
    # colisões de injeção em rajada) tornam a detecção por período não confiável. Ligar
    # apenas para experimentos, fornecendo expected_cycle da spec. Ver docs/resultados-fase2.md.
    check_interval: bool = False
    # ciclo esperado por Message ID (s), da spec/docx. Sobrepõe o ajuste por mediana.
    # Ex.: {"0x27590010": 0.040}. Mids ausentes usam o ajuste (best-effort).
    expected_cycle: dict = field(default_factory=dict)

    # estado aprendido
    known_ids_: set = field(default_factory=set, init=False)
    whitelist_: dict = field(default_factory=dict, init=False)  # mid -> {campo: set(valores)}
    cycle_: dict = field(default_factory=dict, init=False)      # mid -> ciclo esperado (s)
    rr_mids_: set = field(default_factory=set, init=False)      # mids Request-Response (têm req+resp)

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "RuleEngine":
        if "message_id" not in df.columns:
            raise ValueError("df precisa da coluna 'message_id' (use load.load_rule_*)")

        counts = df["message_id"].value_counts()
        self.known_ids_ = set(counts[counts >= self.min_mid_count].index)

        self.whitelist_, self.cycle_, self.rr_mids_ = {}, {}, set()
        for mid in self.known_ids_:
            self._fit_mid(df[df["message_id"] == mid], mid)
        return self

    def _fit_mid(self, sub: pd.DataFrame, mid: str) -> None:
        """Aprende whitelist, ciclo e classe RR para um Message ID."""
        n = len(sub)
        wl = {}
        for col in self.static_fields:
            if col in sub.columns:
                vc = sub[col].value_counts(dropna=False)
                wl[col] = set(vc[vc / n >= self.min_freq].index)
        self.whitelist_[mid] = wl

        # ciclo esperado: spec (docx) tem prioridade; senão mediana dos dt positivos
        if mid in self.expected_cycle:
            self.cycle_[mid] = float(self.expected_cycle[mid])
        elif "time" in sub.columns:
            dt = sub.sort_values("time")["time"].diff()
            dt = dt[dt > 0]
            self.cycle_[mid] = float(dt.median()) if len(dt) else np.nan

        # Request-Response: whitelist de message_type contém request E response
        mtypes = wl.get("message_type", set())
        if MSG_REQUEST in mtypes and MSG_RESPONSE in mtypes:
            self.rr_mids_.add(mid)

    # -------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna DataFrame (mesmo índice) com: anomaly, attack_type, rules."""
        pos = {idx: i for i, idx in enumerate(df.index)}
        rules_hit = [[] for _ in range(len(df))]

        self._rule_static(df, rules_hit)
        self._rule_return_code(df, rules_hit)
        if self.check_state:
            self._rule_state(df, rules_hit, pos)
        if self.check_interval:
            self._rule_interval(df, rules_hit, pos)

        attack_type = [self._classify(h) for h in rules_hit]
        anomaly = [bool(h) for h in rules_hit]
        return pd.DataFrame(
            {"anomaly": anomaly, "attack_type": attack_type, "rules": rules_hit},
            index=df.index,
        )

    # ----- regras individuais (cada uma anexa rótulos em rules_hit) -----
    def _rule_static(self, df, rules_hit):
        """R_UNKNOWN_ID + R_STATIC: Message ID e campos estáticos na whitelist."""
        for i, (_, row) in enumerate(df.iterrows()):
            mid = row["message_id"]
            if mid not in self.known_ids_:
                rules_hit[i].append("R_UNKNOWN_ID")
                continue
            for col, allowed in self.whitelist_[mid].items():
                if row[col] not in allowed:
                    rules_hit[i].append("R_STATIC")
                    break

    @staticmethod
    def _rule_return_code(df, rules_hit):
        """R_RETURN_CODE: return_code de erro -> processo anormal."""
        if "return_code" not in df.columns:
            return
        for i, bad in enumerate(~df["return_code"].isin(VALID_RETURN_CODES)):
            if bad:
                rules_hit[i].append("R_RETURN_CODE")

    def _rule_state(self, df, rules_hit, pos):
        """R_STATE: missing request/response em RPC-RR, pareando por (client_id, session_id)."""
        sub = df[df["message_id"].isin(self.rr_mids_)]
        if sub.empty:
            return
        gb = sub.groupby(["message_id", "client_id", "session_id"])["message_type"]
        has_req = gb.transform(lambda s: (s == MSG_REQUEST).any())
        has_resp = gb.transform(lambda s: (s == MSG_RESPONSE).any())
        missing = (has_req & ~has_resp) | (has_resp & ~has_req)
        for idx in sub.index[missing]:
            rules_hit[pos[idx]].append("R_STATE")

    def _rule_interval(self, df, rules_hit, pos):
        """R_INTERVAL (DoS, best-effort): intervalo abaixo do ciclo esperado."""
        if "time" not in df.columns:
            return
        order = df.sort_values("time")
        for mid in self.known_ids_:
            cyc = self.cycle_.get(mid, np.nan)
            if not np.isfinite(cyc) or cyc <= 0:
                continue
            dt = order[order["message_id"] == mid]["time"].diff()
            thr = self.interval_factor * cyc
            for idx, d in dt.items():
                if pd.notna(d) and d < thr:  # inclui dt=0 (injeção no mesmo timestamp)
                    rules_hit[pos[idx]].append("R_INTERVAL")

    @staticmethod
    def _classify(hits) -> str:
        """Prioridade: Fuzzy > DoS > Abnormal > normal."""
        if "R_UNKNOWN_ID" in hits or "R_STATIC" in hits:
            return "fuzzy"
        if "R_INTERVAL" in hits:
            return "dos"
        if "R_RETURN_CODE" in hits or "R_STATE" in hits:
            return "abnormal"
        return "normal"

    # --------------------------------------------------------------- report
    @staticmethod
    def report(pred: pd.DataFrame) -> dict:
        counts = pred["attack_type"].value_counts().to_dict()
        total = len(pred)
        anomalies = int(pred["anomaly"].sum())
        return {
            "total": total,
            "normal": counts.get("normal", 0),
            "fuzzy": counts.get("fuzzy", 0),
            "dos": counts.get("dos", 0),
            "abnormal": counts.get("abnormal", 0),
            "anomalies_total": anomalies,
        }


if __name__ == "__main__":
    from src.data import load

    # --- Arquivo SOME/IP (header de evento/RPC) ---
    df_someip = load.load_rule_someip()
    eng1 = RuleEngine(static_fields=SOMEIP_STATIC).fit(df_someip)
    rep1 = eng1.report(eng1.predict(df_someip))

    # --- Arquivo Service Discovery ---
    df_sd = load.load_rule_sd()
    eng2 = RuleEngine(static_fields=SD_STATIC).fit(df_sd)
    rep2 = eng2.report(eng2.predict(df_sd))

    comb = {k: rep1.get(k, 0) + rep2.get(k, 0) for k in rep1}
    print("--- SOME/IP ---", rep1)
    print("--- SD      ---", rep2)
    print("--- COMBINADO (comparar c/ Tabela 7: total 144574 | normal 55010 |"
          " fuzzy 43867 | dos 12188 | abnormal 33509) ---")
    for k, v in comb.items():
        print(f"  {k:18s}: {v}")
    print("\nNota: DoS (intervalo) desligado por padrão — ver docs/resultados-fase2.md.")
