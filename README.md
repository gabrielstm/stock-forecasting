# Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction
# Stock Forecasting — atenção/CNN-LSTM e XGBoost (híbrido)

Este repositório contém código para treinar e avaliar um modelo híbrido de previsão de preços de ações (modelos base: ARIMA para resíduos, e um modelo neural com atenção/CNN-LSTM). O script principal é `Main.py`, que prepara os dados, treina o modelo e gera previsões.

## Requisitos

- Python 3.8+ (testado com 3.12 no ambiente atual)
- Pip
- Dependências listadas em `requirements.txt` (NumPy, Pandas, scikit-learn, TensorFlow, Matplotlib, XGBoost, statsmodels, etc.)

## Instalação (Windows / PowerShell)

1) Crie e ative um ambiente virtual (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Instale as dependências:

```powershell
pip install -r requirements.txt
```

Se preferir chamar um executável Python específico (por exemplo quando há mais de uma versão instalada), use o caminho completo e o operador `&` do PowerShell:

```powershell
& "C:/Program Files/Python312/python.exe" -m venv .venv
& "C:/Program Files/Python312/python.exe" -m pip install -r requirements.txt
```

## Como executar

Com o ambiente virtual ativado:

```powershell
python Main.py
```

Se quiser chamar diretamente um Python fora do PATH (obs.: as aspas exigem o operador `&` no PowerShell):

```powershell
& "C:/Program Files/Python312/python.exe" Main.py
```

Observação: o `Main.py` treinará um modelo (podendo levar tempo) e exibirá gráficos com Matplotlib. Ao final ele salva `stock_model.h5` e `stock_normalize.npy` no diretório do projeto.

### Outros scripts disponíveis

- `nlinear.py`: baseline totalmente conectado (N-BEATS/NLinear style). Aceita flags como `--epochs`, `--batch-size`, `--time-steps`, `--learning-rate` e `--no-plot`.
    ```powershell
    python nlinear.py --epochs 100 --batch-size 32
    ```
- `patchTST.py`: implementação em TensorFlow do PatchTST (Transformer com patches temporais). Além das flags anteriores, expõe hiperparâmetros como `--patch-len`, `--patch-stride`, `--d-model`, `--num-heads`, `--layers` e `--dropout`.
    ```powershell
    python patchTST.py --epochs 100 --batch-size 32 --patch-len 4 --patch-stride 2
    ```
- `LSTM.py`: roda as variantes LSTM (residual e série completa) e gera gráficos/resultados consolidando a etapa baseline.

Todos usam os mesmos dados (`601988.SH.csv` e `ARIMA_residuals1.csv`) para treino/teste e exibem métricas de avaliação. Use `--no-plot` para pular a exibição dos gráficos em execuções automatizadas (mesmo com `--no-plot`, os arquivos são salvos).

### Resultados salvos automaticamente

Cada script salva suas saídas em `results/`:

- `*_predictions.png` e `lstm_*_loss.png`: gráficos de treino/predição com o nome do modelo.
- `*_test_results.txt`: arquivo texto com métricas (MSE, RMSE, MAE, R²) seguido de toda a tabela de comparação (`date`, `true`, `pred`, `abs_error`).

Exemplo de execução headless e coleta de artefatos:

```powershell
python nlinear.py --epochs 50 --batch-size 32 --no-plot
python patchTST.py --epochs 50 --batch-size 32 --no-plot
python LSTM.py
```

## Arquivos de entrada esperados

Coloque os CSVs necessários no diretório do projeto (ou ajuste os caminhos no `Main.py`):

- `601988.SH.csv`
- `ARIMA_residuals1.csv`

## Saída

- `stock_model.h5` — modelo salvo
- `stock_normalize.npy` — parâmetros de normalização salvos

## Dicas e solução de problemas

- Erro do PowerShell: "Token 'Main.py' inesperado na expressão ou instrução": use `& "caminho\python.exe" Main.py` para executar quando o caminho tem espaços.
- Se faltar pacote, rode `pip install -r requirements.txt` ou `pip install <pacote>`.
- Se o TensorFlow reportar problemas com GPU, confira a versão do CUDA/cuDNN compatível ou use a versão CPU-only do TF.
- Para executar apenas predição com um modelo salvo (sem treinar), edite `Main.py` para pular o bloco de treinamento ou extraia a função de predição em `utils.py` (posso ajudar a criar um script `predict.py` se quiser).

## Referências / Citation

O trabalho base e arquiteturas relacionadas estão descritas em artigos de atenção + LSTM e híbridos com XGBoost — consulte o artigo abaixo como referência se desejar citar a implementação original:

```
@article{shi2022attclx,
    author={Zhuangwei Shi and Yang Hu and Guangliang Mo and Jian Wu},
    title={Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction},
    journal={arXiv preprint arXiv:2204.02623},
    year={2022},
}
```

---

Se quiser, eu adiciono um pequeno `run.ps1` que automatiza criar/ativar o venv e executar `Main.py`, ou um `predict.py` para usar apenas o modelo salvo. Diga qual prefere.
