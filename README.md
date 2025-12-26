# An Evaluation of ARIMA-Based Deep Learning Models for Stock Price Forecasting in the Brazilian Financial Market (B3)

This repository contains code to train and evaluate different models for stock price forecasting in the Brazilian financial market (B3). Each evaluated model is implemented in a separate .py file.

## Requirements

- Python 3.8+ (tested with Python 3.12 in the current environment)
- Pip
- Dependencies listed in requirements.txt (NumPy, Pandas, scikit-learn, TensorFlow, Matplotlib, XGBoost, statsmodels, etc.)

## Installation (Windows / PowerShell)

1) Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install the dependencies:

```powershell
pip install -r requirements.txt
```

If you prefer to use a specific Python executable (e.g., when multiple versions are installed), use the full path and the PowerShell & operator:

```powershell
& "C:/Program Files/Python312/python.exe" -m venv .venv
& "C:/Program Files/Python312/python.exe" -m pip install -r requirements.txt
```

## How to run

With the virtual environment activated:

```powershell
python modelname*.py
```

If you want to directly call a Python executable outside the PATH (note: quotes require the & operator in PowerShell):

```powershell
& "C:/Program Files/Python312/python.exe" modelname*.py
```

Note: At the end of execution, the results are plotted in a graph and the evaluation metrics are printed in the terminal.

### Available Scripts

For the following models and their variations, the same set of command-line flags can be used:

- `nlinear.py`: fully connected baseline model (N-BEATS / NLinear style). Accepts flags such as --epochs, --batch-size, --time-steps, --learning-rate, and --no-plot.
    ```powershell
    python nlinear.py --epochs 100 --batch-size 32
    ```
- `patchTST.py`: TensorFlow implementation of PatchTST (Transformer with temporal patches). In addition to the previous flags, it exposes hyperparameters such as --patch-len, --patch-stride, --d-model, --num-heads, --layers, and --dropout.
    ```powershell
    python patchTST.py --epochs 100 --batch-size 32 --patch-len 4 --patch-stride 2
    ```

## Expected Input Files

Place the required CSV files in the project root directory (or adjust the paths directly in modelname*.py):

- `historico_b3_indicadores.csv`
- `ARIMA_residuals1.csv`

Note: Running ARIMA.py will generate ARIMA_residuals1.csv, which is equivalent to the version provided in the ./data directory.

## Output

Each script automatically saves its outputs in the ./results directory:

- `*_predictions.png` and `lstm_*_loss.png`: training and prediction plots named after the model.
- `*_test_results.txt`: text file containing evaluation metrics (MSE, RMSE, MAE, R²), followed by the full comparison table (`date`, `true`, `pred`, `abs_error`).

## Tips and Troubleshooting

- If a required package is missing, run pip install -r requirements.txt or pip install <package>.

- If TensorFlow reports GPU-related issues, verify the compatible CUDA/cuDNN versions or use the CPU-only TensorFlow build.

## References / Citation

The present study is inspired by the methodology proposed in the following work:

```
@article{shi2022attclx,
    author={Zhuangwei Shi and Yang Hu and Guangliang Mo and Jian Wu},
    title={Attention-based CNN-LSTM and XGBoost hybrid model for stock prediction},
    journal={arXiv preprint arXiv:2204.02623},
    year={2022},
}
```

---