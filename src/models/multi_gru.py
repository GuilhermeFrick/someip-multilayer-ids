"""Camada 2 — Modelo multi-GRU (Seção 4.6 do artigo).

Uma GRU empilhada (depth=2) **por Message ID**; as saídas ocultas finais são
concatenadas e passam por uma camada linear + softmax -> {Normal, Tamper, Replay}.

Arquitetura (Tabela 4 do artigo), com hidden por ID = x_id * hscale:
    GRU_id (2 camadas)  : entrada (B, l_id, x_id)   -> h_n[-1] (B, x_id*hscale)
    Concatenação        : (B, sum(x)*hscale)
    Linear              : (B, 3)

Entrada do forward (vinda de preprocess.build_ai_dataset):
    X : (B, L, 6)  sinais normalizados (L=91)
    M : (B, L)     índice do Message ID de cada pacote (0..n_ids-1)

O roteamento dos pacotes por ID é vetorizado (sem laço em Python por amostra).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..data.load import SIGNALS_PER_ID, EVENT_IDS


def _gather_by_id(X: torch.Tensor, M: torch.Tensor, j: int, x_id: int):
    """Extrai, por amostra, os pacotes do Message ID índice `j`, preservando a ordem.

    Retorna (Xs, lengths):
      Xs      : (B, Lmax, x_id) sub-sequências (zero-padded à direita)
      lengths : (B,) nº de pacotes do ID por amostra (clamped >=1)
    """
    B, L, F = X.shape
    mask = M == j                                   # (B, L)
    lengths = mask.sum(dim=1)                        # (B,)
    ar = torch.arange(L, device=X.device).expand(B, L)
    key = torch.where(mask, ar, ar + L)              # selecionados primeiro, em ordem
    order = key.argsort(dim=1)                        # (B, L)
    Xs = torch.gather(X, 1, order.unsqueeze(-1).expand(B, L, F))
    Lmax = int(max(int(lengths.max().item()), 1))
    Xs = Xs[:, :Lmax, :x_id].contiguous()            # só os sinais válidos do ID
    return Xs, lengths.clamp(min=1)


class MultiGRU(nn.Module):
    def __init__(self, hscale: int = 5, num_classes: int = 3,
                 signals_per_id: dict | None = None, depth: int = 2):
        super().__init__()
        self.signals_per_id = dict(signals_per_id or SIGNALS_PER_ID)
        self.ids = list(self.signals_per_id.keys())
        self.hscale = hscale

        self.grus = nn.ModuleList([
            nn.GRU(input_size=x, hidden_size=x * hscale,
                   num_layers=depth, batch_first=True)
            for x in self.signals_per_id.values()
        ])
        concat_dim = sum(self.signals_per_id.values()) * hscale
        self.linear = nn.Linear(concat_dim, num_classes)

    def forward(self, X: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        outs = []
        for j, (x_id, gru) in enumerate(zip(self.signals_per_id.values(), self.grus)):
            Xs, lengths = _gather_by_id(X, M, j, x_id)
            packed = pack_padded_sequence(
                Xs, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = gru(packed)          # h_n: (depth, B, x_id*hscale)
            outs.append(h_n[-1])          # última camada -> (B, x_id*hscale)
        cat = torch.cat(outs, dim=1)      # (B, sum(x)*hscale)
        return self.linear(cat)           # logits (B, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # smoke test sem dados reais
    torch.manual_seed(0)
    B, L = 8, 91
    n_ids = len(EVENT_IDS)
    X = torch.randn(B, L, 6)
    M = torch.randint(0, n_ids, (B, L))
    model = MultiGRU(hscale=5)
    out = model(X, M)
    print("saída:", tuple(out.shape), "| parâmetros:", count_parameters(model))
    assert out.shape == (B, 3)
    print("OK")
