import scripts._bootstrap  # noqa: F401
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import calculate_mpt, get_dynamic_bounds, optimize_portfolio_analytical
from app import get_bist30_ml_with_horizon, run_monte_carlo_simulation

print("Testing dynamic bounds for N=5:")
print(get_dynamic_bounds(5))

print("\nRunning get_bist30_ml_with_horizon directly for top 5...")
ml_ret, cl_ret, cl_cov, ml_cov = get_bist30_ml_with_horizon(12)

# Pick 5 arbitrary stocks for testing
top_tickers = ['THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'BIMAS.IS', 'TUPRS.IS']
ml_ret_sliced = ml_ret[top_tickers]
ml_cov_sliced = ml_cov.loc[top_tickers, top_tickers]

ml_max_sliced, ml_min_sliced, ml_graph = run_monte_carlo_simulation(
    ml_ret_sliced.values, ml_cov_sliced.values, top_tickers, "BIST 30 ML (12 Ay - 5 Hisse)",
    months=12
)

print("\n--- Weights ---")
for t in top_tickers:
    print(f"{t}: {ml_max_sliced[t]:.4f}")
