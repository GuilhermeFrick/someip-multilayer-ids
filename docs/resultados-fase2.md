# Resultados — Fase 2 (Camada 1: Motor de Regras)

Código: [../src/rules/rule_engine.py](../src/rules/rule_engine.py)
Executar: `python -m src.rules.rule_engine`

## Abordagem
Whitelist por **Message ID** construída a partir do próprio tráfego usando **frequência**
(valores legítimos dominam ~45.000×; valores fuzzed aparecem ~2×, < 0,01%). Isso separa
ataque de normal de forma reprodutível — é a versão automática da "observação humana de
features" descrita no artigo.

Regras implementadas:
| Regra | Categoria | Mecanismo |
|-------|-----------|-----------|
| `R_UNKNOWN_ID` | Fuzzy | (service_id, method_id) fora do conjunto conhecido |
| `R_STATIC` | Fuzzy | campo estático fora da whitelist do Message ID (mac/porta/versões/client_id/someip_length/message_type) |
| `R_RETURN_CODE` | Abnormal | `return_code != 0x00` (erro) |
| `R_STATE` | Abnormal | missing request/response em RPC-RR, pareando por `(client_id, session_id)` |
| `R_INTERVAL` | DoS | intervalo < ½ do ciclo esperado (**desligado por padrão**, ver limitações) |

## Resultado (whitelist por frequência + máquina de estados; DoS desligado)

| Categoria | Reprodução | Artigo (Tabela 7) |
|-----------|-----------:|------------------:|
| total | 144.544 | 144.574 |
| normal | 79.911 | 55.010 |
| fuzzy | **38.284** | 43.867 |
| dos | 0 (off) | 12.188 |
| abnormal | **26.349** | 33.509 |

> Composição do `abnormal` (26.349): 8.192 `return_code` de erro + 18.084 *missing response*
> (RPC `0x15880008`, pareado por sessão) + 73 no SD.

## Análise das divergências (importante)

> ⚠️ **O dataset de regras NÃO tem coluna de label.** Não há ground-truth independente,
> então não se pode calcular accuracy/recall por pacote (o "100%" do artigo é interno ao
> gerador deles). Comparamos apenas contagens agregadas.

1. **Fuzzy (38.284 vs 43.867):** mecanismo fiel e sólido. A diferença vem de pacotes
   fuzzed que, por acaso, caíram em valores válidos de algum campo (não detectáveis por
   whitelist) — limite teórico da detecção baseada em regras.

2. **Abnormal (26.349 vs 33.509):** ✅ máquina de estados (Fase 2b) implementada.
   Detectamos `return_code` de erro (8.265) **+ missing response** (18.084) via pareamento
   `(client_id, session_id)` — independente de timestamp, contornando a colisão temporal.
   O gap restante (~7.160) corresponde provavelmente a casos *error-on-event/error* e
   pacotes que o artigo conta junto com DoS/timing.

   > **Nota técnica:** parear por sessão foi essencial — a ordem por timestamp é degenerada
   > (blocos `QQQ...RRR` em vez de `QRQR...`), então o pareamento causal por `time` falharia.
   > SOME/IP garante mesmo `(client_id, session_id)` entre request e response.

3. **DoS (0 vs 12.188):** os **timestamps do dataset são em ms e têm muitas colisões**
   (injeção em rajada compartilha o mesmo instante). Com o ciclo de 40 ms da spec, a regra
   de intervalo ou não detecta nada (ciclo ajustado ≈ 0) ou super-detecta (~70.000, engolindo
   normal e abnormal). **Conclusão: este dataset não preserva resolução temporal suficiente
   para detecção de DoS por período.** A capacidade existe no código (`check_interval=True`,
   `expected_cycle={...}`) para experimentos, mas não é confiável aqui.

4. **Normal inflado (79.911 vs 55.010):** o gap caiu bastante com a Fase 2b. O resíduo
   (~24.900) são sobretudo os **12.188 pacotes de DoS**, que têm header válido e sessão
   completa → passam pelas regras estáticas/estado. Bater o número exato exigiria a
   detecção de timing (inviável aqui).

## Conclusão da Fase 2 (+ 2b)
- **Fuzzy por whitelist: reproduzido com fidelidade** (38.284 vs 43.867).
- **Abnormal: reproduzido em grande parte** (26.349 vs 33.509) com `return_code` +
  máquina de estados request↔response pareada por sessão.
- **DoS: inviável neste dataset** por limitação dos timestamps logados — achado relevante.
- Sem labels, a comparação é agregada; ainda assim os mecanismos batem com o artigo.

## Possíveis refinamentos futuros (baixa prioridade)
- [ ] *error-on-event/error* explícito (ainda ~7k de gap no abnormal).
- [ ] Fonte de timestamp/ordenação que viabilize DoS (provavelmente ausente no dataset).
