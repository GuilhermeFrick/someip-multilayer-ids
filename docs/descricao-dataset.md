# Descrição do Dataset SOME-IP-IDS

**Fonte:** https://github.com/yzyGo/Dataset-for-SOME-IP-IDS (ref. [50] do artigo).
**Local:** [../data/raw/](../data/raw/)
**Texto-fonte:** [../references/descricao_dataset.txt](../references/descricao_dataset.txt)

---

## Arquitetura E/E simulada (cenário ACC)

| Nó | IP | Papel |
|----|----|-------|
| **HPC** | 192.168.1.2 | Roda o algoritmo ACC (calcula throttle, frenagem, velocidade relativa) |
| **ADAS** | 192.168.1.1 | Envia dados de câmera/sensores (distância, Doppler, ângulo TIS1/TIS2) |
| **Body/Dynamic** | 192.168.1.3 | Envia cinemática do veículo; recebe sinais de freio/acelerador |
| **GUI** | 192.168.1.4 | Dispara assinatura do serviço ACC e exibe resultados |

## Serviços SOME/IP

| Service ID | Nome | Publica | Assina |
|---|---|---|---|
| `0x1472` | ACC service | HPC | GUI / Body / Dynamic |
| `0x2759` | ADAS service | ADAS | HPC |
| `0x3612` | Vehicle information | Body/Dynamic | HPC |
| `0x1588` | Air conditioning RPC | GUI | HPC |

## Message IDs (Service ID + Method ID) e payload

| Message ID | Nº sinais | Tipo | Ciclo | Descrição dos sinais |
|---|---|---|---|---|
| `0x14720011` | 2 | Event | 10 ms | BrakePres, Throttle |
| `0x14720012` | 2 | Event | 10 ms | leadcar_speed, HWT (collision warning time) |
| `0x27590010` | 6 | Event | 40 ms | Range_1, DopplerV_1, Degree_1, Range_2, DopplerV_2, Degree_2 |
| `0x36120009` | 3 | Event | 10 ms | V, yraw, roty (velocidade, heading, rotação eixo-y) |
| `0x15880008` | 1 | RPC Request-Response | — | Set temperatura A/C |
| `0x15880007` | 1 | RPC Fire&Forget | — | Liga A/C |

> Payload físico = `8 × 6` bytes (6 doubles IEEE 754) por mensagem de evento; só parte é "válida".
> Nos CSVs, isso **já vem deserializado** em `signal1..signal6` (float).

---

## Parte 1 — `data/raw/ai_detection/` (Camada 2 / multi-GRU)

Atua no **payload**. **12 CSVs = 4 cenários × 3 condições.**

**Cenários (Prescan):**
1. `straight_constant_speed` — líder em reta, velocidade constante
2. `straight` — reta com acelerações segmentadas
3. `jam` — congestionamento (líder arranca/para)
4. `bend` — reta + curva, velocidades variadas

**Condições (sufixo):** `_n` = normal · `_t` = tamper · `_r` = replay

**Colunas:**
```
(index), time, mac_dst, mac_src, ipv4_protocol, srcport, dstport,
service_id, method_id, someip_length, client_id, session_id,
protocol_version, interface_version, message_type, return_code,
signal1, signal2, signal3, signal4, signal5, signal6, label
```

**Labels:** `0` = Tamper · `1` = Normal · `2` = Replay

**Tipos de ataque (scripts Python):**
- Tamper: linear / random / fixed-value
- Replay: single / zone

**Volume real (linhas, contagem própria):**

| Cenário | normal | tamper (0) | replay (2) | total/cenário* |
|---|---|---|---|---|
| bend | 71.666 | 21.450 | 37.258 | — |
| jam | 147.879 | 44.250 | 76.882 | — |
| straight | 343.678 | 103.050 | 178.698 | — |
| straight_const | 263.501 | 79.050 | 137.020 | — |

\* Cada arquivo `_n/_t/_r` repete a base normal; somando tudo ≈ **2.480.172 amostras** (= "original data samples" do artigo).

> Sub-ataques (linear/random/fixed, single/zone) **não** têm coluna própria — o label é só 0/1/2.
> Para reproduzir os experimentos por sub-tipo seria preciso re-derivar ou tratar como um só.

---

## Parte 2 — `data/raw/rule_detection/` (Camada 1 / regras)

Atua no **header**. **2 CSVs.**

### `SOMEIPHeader_rule_someip.csv` — comunicação normal
Mesmas colunas da parte de IA, mas **o payload é irrelevante** (regra só olha header).

### `SOMEIPHeader_rule_SD.csv` — Service Discovery
Header **+ campos do SD** (arrays de entries/options):
```
... is_sd, sd_reboot_flag, sd_unicast_flag, sd_entry_lenth,
sd_entry1_type, sd_entry1_index_option1, sd_entry1_index_option2,
sd_entry1_num_option1, sd_entry1_num_option2, sd_entry1_sid,
sd_entry1_instance_id, sd_entry1_major_version, sd_entry1_ttl,
sd_entry1_minor_version, sd_option_length, sd_option1_length,
sd_option1_type, sd_option1_ipv4, sd_option1_l4, sd_option1_port,
sd_entry1_counter, sd_entry1_eventgroup_id
```

**Distribuição (docx = Tabela 7 do artigo):**

| Total | Normal | Fuzzy | DoS | Proc. anormal |
|---|---|---|---|---|
| 144.574 | 55.010 | 43.867 | 12.188 | 33.509 |

**Tipos de ataque no header:**
- **Fuzzy** — varre/altera valores dos campos do header
- **DoS** — reduz o período dos pacotes normais
- **Proc. anormal** — `return_code` de erro, ou perda de response/request

---

## Mapa: dataset → camadas do IDS

```
Pacote SOME/IP
  │
  ▼  CAMADA 1 (regras)  ── usa data/raw/rule_detection/
  │     someip.csv → header de evento/RPC
  │     SD.csv     → arrays do Service Discovery
  │     detecta: Fuzzy, DoS, processo anormal
  │
  ▼  CAMADA 2 (multi-GRU)  ── usa data/raw/ai_detection/
        signal1..6 = payload deserializado
        detecta: Tamper(0), Replay(2) vs Normal(1)
```

## Notas práticas para reprodução
1. **Pré-processamento parcial já feito:** payload já é `signal1..6` em float. Falta **normalizar (min-max por Message ID)** e **janelar (len=91, step=30)**.
2. **Message ID** = `service_id` + `method_id` (ex.: `0x2759` + `0x0010` → `0x27590010`).
3. Camada 1: regras `if/else` sobre IP/MAC/porta/versões/`message_type`/`session_id`/intervalo de `time`; whitelist sai do *System service definition table*.
4. **Desbalanceamento:** arquivos `_n/_t/_r` separados; combinar e ajustar proporção normal/ataque (ratios testados: 40%/20%/1%).
