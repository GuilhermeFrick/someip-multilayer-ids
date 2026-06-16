# Reprodução: Multi-Layer IDS para SOME/IP

Projeto de mestrado para **reproduzir** o trabalho:

> Luo, F.; Yang, Z.; Zhang, Z.; Wang, Z.; Wang, B.; Wu, M.
> *A Multi-Layer Intrusion Detection System for SOME/IP-Based In-Vehicle Network.*
> **Sensors 2023, 23, 4376.** https://doi.org/10.3390/s23094376

O artigo propõe um IDS de **duas camadas** para redes veiculares baseadas em SOME/IP:
1. **Camada baseada em regras** — atua no header SOME/IP, mensagens SD, intervalo e processo de comunicação (detecta Fuzzy, DoS, processo anormal).
2. **Camada baseada em IA (multi-GRU)** — atua no payload dos eventos (detecta Spoof: Tamper e Replay).

> ⚠️ **Os autores NÃO publicaram o código do IDS** — apenas o dataset. Esta reprodução
> reimplementa as duas camadas a partir da descrição do artigo. Ver
> [docs/plano-reproducao.md](docs/plano-reproducao.md).

## Estrutura do projeto

```
LUO/
├── README.md                      # este arquivo
├── requirements.txt               # dependências Python
├── .gitignore
│
├── docs/                          # documentação do projeto
│   ├── analise-artigo.md          # como o trabalho original foi feito
│   ├── descricao-dataset.md       # estrutura e semântica do dataset
│   └── plano-reproducao.md        # plano de reimplementação (roadmap)
│
├── references/                    # material-fonte (não editar)
│   ├── artigo_luo_2023.pdf        # artigo original
│   ├── artigo_luo_2023.txt        # texto extraído do PDF
│   ├── descricao_dataset.txt      # texto extraído do docx
│   └── description_of_dataset.docx
│
├── data/
│   ├── raw/                       # dataset original (yzyGo/Dataset-for-SOME-IP-IDS)
│   │   ├── ai_detection/          # 12 CSVs (4 cenários × {n,t,r})
│   │   └── rule_detection/        # 2 CSVs (someip header + SD)
│   └── processed/                 # sequências/normalizações geradas (output)
│
├── src/                           # código da reprodução
│   ├── data/                      # carga e pré-processamento
│   ├── rules/                     # motor de regras (Camada 1)
│   └── models/                    # modelo multi-GRU (Camada 2)
│
├── notebooks/                     # exploração e prototipagem
├── models/                        # pesos treinados (output)
└── results/                       # métricas, figuras, matrizes de confusão
```

## Dataset

Fonte: https://github.com/yzyGo/Dataset-for-SOME-IP-IDS (referência [50] do artigo).
Já está em `data/raw/`. Detalhes em [docs/descricao-dataset.md](docs/descricao-dataset.md).

## Status

- [x] Análise do artigo
- [x] Análise e documentação do dataset
- [x] Estrutura do projeto
- [x] Carga + pré-processamento (`src/data/load.py`, `preprocess.py` + notebook demo)
- [x] Camada 1 — motor de regras (Fuzzy sólido; Abnormal c/ máquina de estados; DoS limitado — ver [docs/resultados-fase2.md](docs/resultados-fase2.md))
- [x] Camada 2 — multi-GRU: **94,61%** accuracy, replay recall 91,5% (arquitetura/seqs idênticas ao artigo; gap p/ 99,78% = hiperparâmetros/épocas — ver [docs/resultados-fase3.md](docs/resultados-fase3.md))
- [x] Avaliação (Accuracy/Precision/Recall/F1 + matriz de confusão)
- [x] Baseline single-GRU: **97,66%** (reproduz o artigo 97,40%) e **supera o multi-GRU (94,92%)** — a tese central do artigo NÃO se reproduz; ver [docs/resultados-fase3.md](docs/resultados-fase3.md)
- [ ] Otimização Bayesiana própria (testar se reabilita a vantagem do multi-GRU)
- [ ] Análise de desbalanceamento (ratios 40%/20%/1%)
