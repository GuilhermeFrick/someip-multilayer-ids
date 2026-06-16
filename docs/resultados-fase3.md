# Resultados — Fase 3 (Camada 2: multi-GRU)

Código: [../src/models/multi_gru.py](../src/models/multi_gru.py) · [../src/models/train.py](../src/models/train.py)
Executar: `python -m src.models.train` (cenário `bend`) ou `train.run(scenarios=[...], epochs=...)`.

## Validação da arquitetura ✅
O modelo multi-GRU implementado tem **13.698 parâmetros** — **idêntico** à Tabela 11 do
artigo ("Multi-GRU: 13,698"). Forte evidência de que a arquitetura (1 GRU empilhada depth=2
por Message ID, hidden = x_id·hscale, concatenação → linear → softmax) está correta.

## Estrutura do ataque de Replay (achado importante)
Inspecionando os arquivos `_r`: os blocos de replay têm **exatamente 8 pacotes** e
**repetem literalmente os valores recentes** da própria série:

```
idx  label  signal1     (0x36120009)
 1     1    24.497593   <- valores normais
 2     1    24.495005
 3     1    24.491268
 4     2    24.500000   <- replay começa (eco do buffer)
 5     2    24.497593   = idx1
 6     2    24.495005   = idx2
 7     2    24.491268   = idx3
```

➡️ Os valores replayed são **normais válidos**, apenas fora da ordem temporal. Por isso
detectá-los exige um modelo que aprenda a **dinâmica natural** da série e flagre o
"rebobinar" — é o caso difícil que o artigo diz que o single-GRU erra e o multi-GRU acerta.

## Resultado preliminar (cenário `bend`, 30 épocas)

| Classe | Precision | Recall | F1 |
|--------|----------:|-------:|----:|
| Tamper | 99,15% | 98,32% | 0,987 |
| Normal | 49,49% | 90,79% | 0,641 |
| Replay | 47,56% | **8,18%** | 0,140 |

**Accuracy: 65,78%** · loss travou em ~0,49 (não convergiu p/ ~0).

Matriz de confusão (linhas=verdadeiro; Tamper/Normal/Replay):
```
[469,   8,   0]   # Tamper: ok
[  1, 434,  43]   # Normal: ok
[  3, 435,  39]   # Replay -> classificado como Normal (435/477)
```

### Interpretação
Este resultado **reproduz o modo de falha do single-GRU** descrito no artigo
("serious misjudgments between replay and normal"), **não** o sucesso do multi-GRU.
O Tamper (valores fora de distribuição) é trivial; o Replay (valores em distribuição,
fora de ordem) não foi aprendido com dados/épocas insuficientes.

Hipótese: o artigo treinou com **todos os 4 cenários (~82.625 sequências)** até a
convergência (<60 épocas). Usamos só `bend` (7.158 seq) e 30 épocas.

## Resultado final (4 cenários, 60 épocas) ✅

Pré-processamento gerou **82.641 sequências** (≈ 82.625 do artigo — outra validação).

| Classe | Precision | Recall | F1 |
|--------|----------:|-------:|----:|
| Tamper | 99,98% | 99,89% | 0,999 |
| Normal | 91,60% | 92,42% | 0,920 |
| Replay | 92,26% | **91,52%** | 0,919 |

**Accuracy: 94,61%** · loss final 0,128 (ainda descendo lentamente).

Matriz de confusão (linhas=verdadeiro; Tamper/Normal/Replay):
```
[5500,    1,    5]   # Tamper: quase perfeito
[   0, 5095,  418]   # Normal
[   1,  466, 5043]   # Replay: agora detectado (recall 91,5%)
```

### Interpretação — reprodução bem-sucedida (qualitativa)
Com os dados completos, o multi-GRU **inverteu o modo de falha**: o replay recall foi de
**8,2% → 91,5%**. Isso reproduz a **vantagem central do multi-GRU sobre o single-GRU**
afirmada no artigo (separar replay de normal). A confusão residual (418 normal→replay,
466 replay→normal) é o que ainda nos separa dos 99,78% do artigo.

| Métrica | Reprodução | Artigo |
|---------|-----------:|-------:|
| Accuracy | 94,61% | 99,78% |
| Parâmetros | 13.698 | 13.698 |
| Sequências | 82.641 | 82.625 |

### Por que ainda não é 99,78% (gap de ~5 pp)
1. **Hiperparâmetros:** usamos os valores **finais** da Tabela 9 (otimizados pelo artigo
   para o pipeline *deles*). Não rodamos nossa própria **otimização Bayesiana** (Tabela 5).
2. **Convergência:** a loss em 60 épocas ainda caía (0,128, não ~0). Mais épocas devem ajudar.
3. **Escolhas de janelamento/rotulagem** (passo 30, rótulo `any_attack`) podem diferir das do artigo.

## Comparação multi-GRU vs single-GRU (Fase 3b) — achado central

Treinamos o **single-GRU** (baseline do artigo, hscale=31, Tabela 9) no mesmo pipeline,
e re-treinamos o multi-GRU por 120 épocas (convergência).

| Modelo (reprodução) | Accuracy | Replay recall | Loss final | Artigo |
|---------------------|---------:|--------------:|-----------:|-------:|
| **single-GRU** | **97,66%** | 94,77% | 0,059 | 97,40% ✓ |
| **multi-GRU** (60 ép) | 94,61% | 91,52% | 0,128 | 99,78% |
| **multi-GRU** (120 ép) | 94,92% | 88,89% | 0,119 | 99,78% ✗ |

### 🔴 A tese central do artigo NÃO se reproduz
O artigo afirma que o **multi-GRU supera o single-GRU** (99,78% vs 97,40%). Na nossa
reprodução independente acontece o **oposto**: o **single-GRU (97,66%) supera o multi-GRU
(94,92%)**.

Pontos a favor da validade da nossa reprodução:
- **Nosso single-GRU (97,66%) bate quase exato o do artigo (97,40%)** — o pipeline está correto.
- Arquiteturas com nº de parâmetros idêntico ao artigo (13.698 / 9.675≈10.047).
- Pré-processamento gera 82.641 seqs ≈ 82.625 do artigo.

O multi-GRU **não está subtreinado**: a loss estacionou em ~0,119 da época 55 à 120
(oscilando 0,12–0,16). Ele **converge para uma solução pior** que o single-GRU sob os
hiperparâmetros declarados (Tabela 9).

### Hipóteses para a discrepância (honestas)
1. **Hiperparâmetros:** usamos os valores finais da Tabela 9 (otimizados pelo artigo para o
   pipeline *deles*); o lr do multi-GRU (0,00896) é ~2× o do single (0,00433) e causa
   oscilação perto do mínimo. Não rodamos nossa própria otimização Bayesiana.
2. **Detalhes de implementação não publicados** (inicialização, concatenação exata,
   normalização) podem ser decisivos para a vantagem do multi-GRU.
3. **Janelamento/rotulagem** (step=30, `any_attack`) podem diferir das escolhas do artigo.

### Implicação para a dissertação
Este é um resultado de reprodução **relevante**: a contribuição-título do artigo (multi-GRU
> single-GRU) **não é robusta** a uma reimplementação independente seguindo o texto. O
single-GRU, mais simples, reproduz e até supera. Vale destacar isso ao avaliar a relevância
do trabalho.

## Fase 3c — Otimização Bayesiana própria (`src/models/optimize.py`)

Busca Optuna (20 trials × 20 épocas, subamostra 25%, poda) nas faixas da Tabela 5, depois
treino do melhor config em 120 épocas no dataset completo.

**Melhor config:** hscale=5, lr=0,00893, β1=0,973, β2=0,960 — quase idêntico à Tabela 9,
exceto **β1 (0,973 vs 0,934)**.

| Modelo | Accuracy | Replay recall | Loss final |
|--------|---------:|--------------:|-----------:|
| single-GRU | 97,66% | 94,77% | 0,059 |
| multi-GRU (Tabela 9 exata) | 94,92% | 88,89% | 0,119 |
| **multi-GRU (Bayes-otimizado)** | **96,79%** | 94,79% | 0,079 |

### Achados
1. A busca **redescobriu a região de hiperparâmetros do artigo** (hscale=5, lr~0,009) —
   confirma que a Tabela 9 está num bom ótimo.
2. Uma mudança pequena (**β1 0,934→0,973**) fez o multi-GRU saltar **94,9% → 96,79%**: a loss
   quebrou o platô de 0,12 e foi a 0,079. O modelo é **muito sensível ao momento do Adam**.
3. **Alta variância entre rodadas:** com mesmos hscale/lr, o multi-GRU caiu em 94,9% ou 96,8%
   dependendo do β e da dinâmica de treino (escapar ou não do platô de loss 0,12).

### Conclusão refinada (substitui a anterior)
Mesmo **otimizado**, o multi-GRU (96,79%) apenas **empata** com o single-GRU (97,66%) — não o
supera. A vantagem grande alegada no artigo (99,78 vs 97,40) **não se reproduz**. A história
real não é "multi ≫ single", e sim **alta sensibilidade a hiperparâmetros/seed e variância**:
os dois modelos ficam na faixa 95–98%, com o single-GRU consistentemente no topo.

> Para a dissertação: a contribuição-título do artigo não é robusta. Recomenda-se reportar
> média ± desvio sobre múltiplas seeds para ambos os modelos (próximo passo natural).
