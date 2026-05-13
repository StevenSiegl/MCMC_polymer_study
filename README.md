# MCMC Polymer Study: Rosenbluth Algorithm

Monte Carlo Markov Chain (MCMC) simulation of large polymer configurations using the Rosenbluth algorithm.

## Project Overview

This project implements the Rosenbluth algorithm for efficient sampling of polymer configurations in high-dimensional spaces. The algorithm is particularly useful for simulating large polymer systems where standard Monte Carlo methods become inefficient due to steric exclusions and configurational constraints.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for visualization)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
MCMC_polymer_study/
├── mcmc_polymer/          # Main package
│   ├── __init__.py
│   ├── rosenbluth.py      # Core Rosenbluth algorithm
│   ├── polymer.py         # Polymer configuration classes
│   └── utils.py           # Utility functions
├── tests/                 # Test suite
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Output data and results
└── requirements.txt
```

## Usage

```python
from mcmc_polymer import RosenbluthMCMC

# Initialize MCMC sampler
sampler = RosenbluthMCMC(n_monomers=100)

# Run simulation
trajectories = sampler.run(n_steps=10000)
```

## Algorithm Details

The Rosenbluth method:
1. Grows polymer chains segment by segment
2. Assigns weights based on available (non-overlapping) positions
3. Accepts/rejects configurations based on statistical weights
4. Improves efficiency for excluded-volume interactions

## References

- Rosenbluth, M. N., & Rosenbluth, A. W. (1955)
- Escobedo, F. A., et al. (1999)

## License

MIT

## Author

[Your Name]
