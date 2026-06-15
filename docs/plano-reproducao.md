# Plano de Reprodução

Roadmap para reimplementar o IDS multicamada do artigo Luo et al. (2023), já que
**o código original não foi publicado** (só o dataset).

---

## Fase 0 — Setup (feito)
- [x] Extrair e analisar artigo + dataset
- [x] Estruturar projeto (`data/`, `src/`, `docs/`, etc.)
- [ ] Criar ambiente Python (`pip install -r requirements.txt`)

## Fase 1 — Carga e pré-processamento (`src/data/`) ✅
- [x] `load.py`: ler CSVs, construir coluna `message_id = service_id||method_id`.
- [x] Manter `signal1..6` como float (deserialização já vem pronta nos CSVs).
- [x] `preprocess.py`:
  - [x] Normalização **min-max por Message ID** (`fit/apply_minmax_per_id`, guarda min/max).
  - [x] **Janelamento**: sequências `len=91`, `step=30` (`make_windows`).
  - [x] Split treino/teste **80/20** estratificado (`build_ai_dataset`).
- [x] Notebook demo: `notebooks/01-exploracao-preprocessamento.ipynb`.
- [ ] (Opcional) Cachear artefatos em `data/processed/` para datasets grandes.

**Saída:** `build_ai_dataset()` retorna `X (n,91,6)`, `M (n,91)` (índice do Message ID
por pacote, p/ rotear no multi-GRU) e `y (n,)`. Validado no cenário `bend`: 7158 janelas,
3 classes balanceadas.

## Fase 2 — Camada 1: motor de regras (`src/rules/`) ✅ (parcial)
- [x] `rule_engine.py`: whitelist por Message ID via **frequência** (robusta a contaminação).
  - [x] **Estáticas (Fuzzy):** mac/portas/versões/client_id/someip_length/message_type. **Sólido.**
  - [x] **Abnormal:** `return_code != 0x00` **+ máquina de estados req↔response** (Fase 2b)
        pareada por `(client_id, session_id)`. 26.349 vs 33.509 do artigo.
  - [~] **DoS (intervalo):** implementado mas **inviável neste dataset** (timestamps em ms
        com colisões). Desligado por padrão.
- [x] Avaliado em `rule_detection/`; comparado à Tabela 7. Ver
      [resultados-fase2.md](resultados-fase2.md).

**Achado-chave:** o dataset de regras **não tem labels** → sem accuracy/recall por pacote
(o "100%" do artigo é interno). Fuzzy e Abnormal reproduzem bem; DoS é limitado pelos timestamps.

## Fase 3 — Camada 2: multi-GRU (`src/models/`) ✅
- [x] `multi_gru.py` (PyTorch): 1 GRU empilhada (depth=2) por Message ID, concatenação →
      Linear → softmax. **13.698 parâmetros = idêntico ao artigo** (Tabela 11). Roteamento
      vetorizado por ID.
- [x] `train.py`: Adam + cross-entropy; hiperparâmetros da Tabela 9 (multi-GRU).
- [x] Resultado: **94,61% accuracy**, replay recall 8% → **91,5%** (4 cenários, 60 épocas).
      Reproduz a vantagem multi-GRU. Ver [resultados-fase3.md](resultados-fase3.md).
- [ ] **Baseline single-GRU** (hscale=31, lr=0.00433...) — comparação direta.
- [ ] (Opcional) **Otimização Bayesiana** (`optuna`) p/ fechar o gap até 99,78%.

## Fase 4 — Avaliação (`src/evaluate.py`, `results/`)
- [ ] Métricas: Accuracy, Precision, Recall, F1, **AUC**, matriz de confusão.
- [ ] Curva de loss (multi vs single-GRU).
- [ ] Tempo de inferência por sequência/pacote (CPU; GPU se disponível).
- [ ] **Metas do artigo:** multi-GRU Acc ≈ 99,78%; single-GRU ≈ 97,40%.

## Fase 5 — Robustez a desbalanceamento
- [ ] Treinar sob ratios anomalia/normal de **40% / 20% / 1%**.
- [ ] Comparar recall de replay (multi-GRU deve resistir; single-GRU deve falhar).

## Fase 6 — Relatório
- [ ] Consolidar resultados vs. artigo; documentar divergências.

---

## Riscos / pontos de atenção
- **Sub-tipos de ataque** (linear/random/fixed; single/zone) não estão rotulados separadamente
  → reprodução fica no nível 0/1/2.
- **Tempo real em hardware embarcado** (Jetson NX) provavelmente fora de escopo;
  focar em corretude de detecção + tempo relativo em CPU/GPU comum.
- **Otimização Bayesiana** é cara; pode-se começar com os hiperparâmetros prontos da Tabela 9.
- Detalhes de implementação ausentes no artigo (ex.: thresholds exatos de intervalo,
  inicialização de pesos) → decisões nossas, documentar.

## Dependências sugeridas
Ver [../requirements.txt](../requirements.txt).
