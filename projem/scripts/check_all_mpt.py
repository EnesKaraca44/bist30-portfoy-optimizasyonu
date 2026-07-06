import importlib.util
import sys
from pathlib import Path

_bootstrap_path = Path(__file__).resolve().parent / "_bootstrap.py"
_spec = importlib.util.spec_from_file_location("_bootstrap", _bootstrap_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

from config import BIST30_TICKERS
from portfolio.optimizer import get_bist30_data_inputs, run_monte_carlo_simulation

ml_ret, cl_ret, cl_cov, ml_cov = get_bist30_data_inputs()

ml_max_sharpe, ml_min_risk, ml_graph = run_monte_carlo_simulation(
    ml_ret, ml_cov, BIST30_TICKERS, "BIST 30 (ML Destekli)"
)

print("ML Max Sharpe Getiri:", ml_max_sharpe["Getiri"])
print("ML Max Sharpe Risk:", ml_max_sharpe["Risk"])
print("ML Max Sharpe Orani:", ml_max_sharpe["SharpeOrani"])
