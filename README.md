# Option Delta-Hedging under Transaction Costs

Simulation and utility-optimal strategy search for a trader holding a
strangle-like option (long call K=90, long put K=107) alongside a
dynamically-adjusted stock position and a risk-free bank account, on a
stock with deterministic but time-varying volatility.

Originally a King's College London coursework project (MSc Applied
Mathematics, 7CCMFM06); rebuilt here as a tested Python library.

## Results

Parameters: `S0=107, mu=0.14, r=0.04, T=1.40, theta=0.01, lambda=0.01, b0=132`,
32 rebalancing steps, 100,000 simulated paths.

| Strategy | q1 | q2 | q3 | phi |
|---|---|---|---|---|
| Do nothing | 0 | 0 | 0 | 1.0576 |
| Buy & hold stock | 1 | 0 | 0 | 1.1500 |
| Delta-hedge | 0 | 1 | 0 | 1.1027 |
| Buy & hold option | 0 | 0 | 1 | 1.1236 |
| **Optimal** | **~0.80** | **~0** | **~1.52** | **1.1790** |

phi is a scaled measure of expected utility; larger is better. The optimal
strategy holds a static stock+option position rather than dynamically
delta-hedging, since under this fee structure the ongoing cost of tracking
the option's delta outweighs the benefit.

## Project structure

src/option_hedging/ the library
tests/ pytest suite
notebooks/results.ipynb worked example reproducing the results above


## Setup

```bash
python3 -m venv hedging-env
source hedging-env/bin/activate
pip install -r requirements.txt
```

Run the tests:
```bash
python3 -m pytest tests/ -v
```

Run the demo:
```bash
jupyter notebook notebooks/results.ipynb
```