# Análise do Artigo — Como o trabalho original foi feito

**Referência:** Luo et al., *A Multi-Layer Intrusion Detection System for SOME/IP-Based In-Vehicle Network*, Sensors 2023, 23, 4376.
**Texto-fonte:** [../references/artigo_luo_2023.txt](../references/artigo_luo_2023.txt)

---

## 1. Motivação / lacuna
- SOME/IP (protocolo de comunicação orientada a serviços sobre Ethernet automotiva) está substituindo o CAN, mas **não tem mecanismo de segurança no AUTOSAR**; TLS/IPsec não encaixam bem.
- Quase **não existiam datasets nem IDS públicos** para SOME/IP.
- O trabalho ataca as duas lacunas: **(a)** método de geração de dataset e **(b)** um IDS multicamada.

## 2. Geração do dataset (porque não existia tráfego real)
Cadeia de simulação:

```
Prescan  →  Simulink  →  CANoe (CAPL)  →  log  →  CSV (via Python)
(cenário    (dinâmica   (encapsula em SOME/IP
 ADAS)       + sinais)   e troca entre nós)
```

- **Prescan**: cenário de estrada (veículos, sensores, trajetória).
- **Simulink**: dinâmica do veículo, gera sinais reais (velocidade, freio, throttle, sensores).
- **CANoe + CAPL**: encapsula em mensagens SOME/IP reais; o **ataque** também é CAPL, disparado por painel (simula backdoor via app).
- Diferencial vs. gerador de terceiros [45]: aqui o **payload tem significado ADAS real** → permite detecção no payload.

## 3. Tipos de ataque definidos
| Ataque | Alvo | Camada que detecta |
|--------|------|--------------------|
| **Fuzzy** | header / arrays SD (adulteração aleatória) | Regras |
| **DoS** | reduzir ciclo de pacotes periódicos | Regras |
| **Processo anormal** | error-on-error/event, missing response/request | Regras |
| **Spoof → Tamper** | payload adulterado (header válido) | IA |
| **Spoof → Replay** | payload reenviado (header válido) | IA |
| Operação não autorizada | assinatura/RPC sem permissão | **Não detectado** (precisa HIDS) |

## 4. Arquitetura do IDS — duas camadas
A ideia central: **não usar IA para tudo**. Filtros em ordem de custo crescente.

### Camada 1 — Baseada em regras (rápida, todo pacote)
- Verifica header SOME/IP, mensagem SD, intervalo e processo de comunicação.
- Funciona como **whitelist**: cada `Message ID` tem um grupo de regras (estáticas, dinâmicas, de estado). Falhou uma → anomalia imediata.
- **Engenharia:** regras codificadas como lógica `if/else` no software (não num "banco de regras") → economiza memória/tempo, ideal para embarcado.

### Camada 2 — Baseada em IA, multi-GRU (só payload de eventos)
- **multi-GRU**: uma GRU empilhada (profundidade 2) **por `Message ID`**; saídas concatenadas → camada linear → softmax → {Normal, Tamper, Replay}.
- **Por que multi-GRU?** Escalabilidade: GRU única explode em parâmetros com o nº de serviços (~500k p/ N=40); multi-GRU cresce linear (~12k) — ~40× menos.

### Pré-processamento (antes da IA)
1. **Deserialização**: payload em hex (IEEE 754, 8 bytes/double) → valor real do sinal.
2. **Normalização**: min-max por `Message ID`.
3. **Janelamento**: sequências de **comprimento 91** (7× msgs/ciclo), **passo deslizante 30**.

### Otimização de hiperparâmetros
- **Otimização Bayesiana** (3-fold CV) para `hscale`, `lr`, β1, β2.
- Otimizador **Adam**, perda **cross-entropy**.

## 5. Implementação e avaliação
- **Plataformas:** laptop i7-8750H (dev) e **Jetson Xavier NX** (embarcado, GPU 384-core Volta, 15 W).
- **Métricas:** Accuracy, Precision, Recall, F1, AUC + **tempo de detecção**.

### Resultados principais
| Item | Resultado |
|------|-----------|
| Regras (acerto) | **100%**; 29,394 µs/pacote |
| multi-GRU (acurácia) | **99,7761%** (Tamper 100%, Replay 99,58%, Normal 99,75%) |
| single-GRU (baseline) | 97,40% — confunde replay/normal |
| Tempo total/evento (NX) | **0,3958 ms** GPU / **0,6669 ms** CPU |
| Vazão | 2526 pkt/s (GPU) / 1499 (CPU) |
| Desbalanceamento | multi-GRU mantém recall replay ~90% a 40/20%; single-GRU falha |

### Hiperparâmetros finais (Tabela 9 do artigo)
| Modelo | hscale | lr | β1 | β2 |
|--------|--------|----|----|----|
| multi-GRU | 5 | 0.0089630704 | 0.933792409392 | 0.952802490181 |
| single-GRU | 31 | 0.0043259137 | 0.939844012507 | 0.943045819607 |

## 6. Equações do GRU (Seção 4.6.1)
```
r_t = σ(W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr)        # reset gate
z_t = σ(W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz)        # update gate
n_t = tanh(W_in·x_t + b_in + r_t ⊙ (W_hn·h_{t-1} + b_hn))   # candidato
h_t = (1 - z_t)⊙n_t + z_t⊙h_{t-1}                     # saída oculta
```

## 7. Resumo em uma frase
> Geraram um **dataset SOME/IP realista por simulação** e, sobre ele, um **IDS de duas camadas** — **regras** para o determinístico/barato (header, SD, intervalo, processo) e **multi-GRU escalável** para o que só a IA pega (payload) — validando em hardware embarcado com foco em tempo real.
