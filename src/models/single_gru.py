"""Baseline single-GRU (comparação da Seção 5.4 / Tabelas 9–11 do artigo).

Ao contrário do multi-GRU, há **uma única GRU empilhada (depth=2)** que processa toda a
sequência de pacotes sem separar por Message ID. Cada passo de tempo é o vetor de 6 sinais
do pacote. Como os sinais de IDs diferentes compartilham as mesmas 6 colunas (significados
físicos distintos), o modelo não distingue bem os IDs — é o motivo pelo qual o artigo mostra
que ele confunde replay com normal, perdendo para o multi-GRU.

Mantém a mesma assinatura de forward (X, M) que o MultiGRU para reusar `train.py`;
o argumento M (índice de Message ID) é ignorado aqui.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SingleGRU(nn.Module):
    def __init__(self, hscale: int = 31, num_classes: int = 3,
                 input_size: int = 6, depth: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hscale,
                          num_layers=depth, batch_first=True)
        self.linear = nn.Linear(hscale, num_classes)

    def forward(self, X: torch.Tensor, M: torch.Tensor | None = None) -> torch.Tensor:
        _, h_n = self.gru(X)          # h_n: (depth, B, hscale)
        return self.linear(h_n[-1])   # logits (B, num_classes)


if __name__ == "__main__":
    from .multi_gru import count_parameters

    torch.manual_seed(0)
    X = torch.randn(8, 91, 6)
    model = SingleGRU(hscale=31)
    out = model(X)
    print("saída:", tuple(out.shape), "| parâmetros:", count_parameters(model))
    assert out.shape == (8, 3)
    print("OK")
