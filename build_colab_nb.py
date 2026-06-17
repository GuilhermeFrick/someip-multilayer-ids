# Gera notebooks/02-reproducao-colab.ipynb — reprodução do IDS de Luo no Colab (GPU)
import json, os

def md(*l): return {"cell_type":"markdown","metadata":{},"source":[x if x.endswith("\n") else x+"\n" for x in l]}
def code(*l): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[x if x.endswith("\n") else x+"\n" for x in l]}

cells = [
md("# Reprodução — IDS Multicamada para SOME/IP (Luo et al., 2023) na GPU",
   "",
   "Reproduz nossa reimplementação independente do IDS de **duas camadas**:",
   "1. **Camada 1 — regras** (Fuzzy / processo anormal) sobre o cabeçalho/SD.",
   "2. **Camada 2 — multi-GRU** (e baseline **single-GRU**) sobre o *payload*.",
   "",
   "**Achado central:** na nossa reimplementação, o **single-GRU iguala/supera o multi-GRU** —",
   "contrariando a contribuição do artigo. Este notebook permite verificar isso na GPU.",
   "",
   "**Use GPU:** Runtime → Change runtime type → GPU (T4 basta)."),

md("## 0. Setup — clonar o repositório (com dados via Git LFS) e instalar deps"),
code("import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"),
code("%cd /content",
    "!git lfs install",
    "![ -d someip-multilayer-ids ] || git clone https://github.com/GuilhermeFrick/someip-multilayer-ids.git",
    "%cd /content/someip-multilayer-ids",
    "!git lfs pull            # baixa os CSVs do dataset (~468 MB)",
    "!pip -q install optuna   # (torch já vem no Colab)",
    "import os; print('dados:', os.path.exists('data/raw/ai_detection/data_bend_4_08_someip_n.csv'))"),

md("## 1. Camada 1 — Motor de regras",
   "",
   "Whitelist por Message ID + máquina de estados request/response. Compara com a Tabela 7 do",
   "artigo (o dataset de regras não tem rótulos, então a comparação é por contagem agregada)."),
code("!python -m src.rules.rule_engine"),

md("## 2. Camada 2 — multi-GRU vs single-GRU",
   "",
   "Treina os dois modelos nos 4 cenários e compara. A arquitetura multi-GRU tem 13.698",
   "parâmetros (idêntico ao artigo). Em GPU cada treino leva poucos minutos."),
code("import sys; sys.path.insert(0, '.')",
    "from src.models import train",
    "SCEN = ['straight_constant_speed','straight','jam','bend']",
    "EPOCHS = 120",
    "res = {}",
    "for mt in ['single','multi']:",
    "    print('='*20, mt+'-GRU', '='*20)",
    "    r = train.run(scenarios=SCEN, epochs=EPOCHS, model_type=mt, verbose=True)",
    "    res[mt] = r['metrics']"),
code("# Comparação lado a lado",
    "import pandas as pd",
    "rows = []",
    "for mt in ['multi','single']:",
    "    m = res[mt]",
    "    rows.append({'Modelo': mt+'-GRU',",
    "                 'Accuracy': round(m['accuracy']*100, 2),",
    "                 'Replay recall': round(m['per_class']['Replay']['recall']*100, 2),",
    "                 'Tamper F1': round(m['per_class']['Tamper']['f1'], 3),",
    "                 'Replay F1': round(m['per_class']['Replay']['f1'], 3)})",
    "pd.DataFrame(rows)"),

md("## 3. (Opcional) Otimização Bayesiana do multi-GRU",
   "",
   "Busca hiperparâmetros (Tabela 5) com Optuna e treina o melhor config. Testa se a vantagem",
   "do multi-GRU reaparece com tuning (na nossa execução em CPU, chegou a 96,8% — ainda abaixo",
   "do single-GRU)."),
code("!python -m src.models.optimize --trials 20 --trial-epochs 20 --subsample 0.5 --final-epochs 120"),

md("---",
   "**Esperado (nossa reprodução):** single-GRU ~97,7% ≥ multi-GRU ~94,9% (mesmo com 120 épocas).",
   "O single-GRU reproduz quase exato o número do artigo (97,40%); o multi-GRU NÃO atinge os",
   "99,78% alegados. Detalhes em `docs/resultados-fase3.md`."),
]

nb = {"cells":cells,"metadata":{"accelerator":"GPU","colab":{"provenance":[]},
      "kernelspec":{"display_name":"Python 3","name":"python3"},
      "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/02-reproducao-colab.ipynb","w",encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("notebook gerado com", len(cells), "celulas")
