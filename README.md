# TCN-GWAO: Temporal Convolutional Network with Genetic Algorithm for RUL Prediction

## Overview

This project implements a **Temporal Convolutional Network (TCN)** combined with a **Genetic Algorithm with Weighted Average Optimization (GWAO)** for Remaining Useful Life (RUL) prediction on aircraft turbofan engines.

The model uses the **CMAPSS dataset** (Commercial Modular Aero-Propulsion System Simulation) to predict how many cycles an engine has left before it needs maintenance.

## Paper Reference

- **Paper**: [Full citation of the paper](https://doi.org/10.1115/1.4064809)

## Dataset

The project uses the **CMAPSS Jet Engine Simulated Data**:
- **Source**: [NASA CMAPSS Dataset](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Description**: Simulated degradation data from jet engines containing sensor readings, operational settings, and failure indicators

## Project Structure

```
TCN-GWAO/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── tcn-gawo.ipynb           # Jupyter notebook with full implementation
├── src/
│   └── main.py              # Main Python script
├── dataset/                 # Data files (not included in repo)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
└── .gitignore              # Git ignore rules
```

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TCN-GWAO.git
cd TCN-GWAO
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CMAPSS dataset from [NASA's repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) and place the files in the `dataset/` directory.

## Usage

### Running with Jupyter Notebook

```bash
jupyter notebook tcn-gawo.ipynb
```

### Running as Python Script

```bash
python src/main.py
```

## Methodology

### 1. Data Processing
- Loads CMAPSS FD001 dataset
- Feature engineering: drops irrelevant columns and applies Min-Max scaling (-1, 1)
- Creates sliding windows of length 10 for temporal sequences
- Defines RUL targets with configurable early RUL threshold

### 2. TCN Model Architecture
```
TCN Layer:
  - Filters: 32
  - Kernel Size: 2
  - Dilations: (1, 2, 4, 8)
  - Padding: Causal
  - Layer Normalization: Enabled
  - Dropout: 0.05

Output Layer:
  - Dense(1) for RUL prediction
```

### 3. Genetic Algorithm Optimization (GWAO)
- **Population Size**: 25 individuals
- **Generations**: 10
- **Selection**: Roulette wheel selection based on fitness
- **Crossover**: Uniform crossover with 50% probability
- **Mutation**: Random weight perturbation with 10% probability
- **Fitness Function**: 1/RMSE (inverted RMSE)

### 4. Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **S-score**: Asymmetric scoring function for RUL prediction

## Key Features

- **Window-based Processing**: Uses sliding windows for temporal dependencies
- **Genetic Algorithm**: Evolves model weights across generations
- **Weighted Averaging**: Combines predictions from multiple test windows
- **Asymmetric Scoring**: S-score metric penalizes early and late predictions differently

## Configuration Parameters

Edit these in `tcn-gawo.ipynb` or `src/main.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `window_length` | 10 | Size of sliding window for sequences |
| `shift` | 1 | Stride for sliding window |
| `early_rul` | 125 | Threshold for early RUL definition |
| `pop_size` | 25 | Genetic algorithm population size |
| `num_generations` | 10 | Number of GA generations |
| `mutation_rate` | 0.1 | Probability of mutation |
| `parent_selection_pressure` | 0.5 | Probability of crossover |

## Results

The model evaluates performance on test data using:
- Average predictions weighted by the number of test windows per engine
- S-score for asymmetric error measurement
- RMSE on both full predictions and last-example-only predictions

## Output Metrics

After training, the model outputs:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- S-score (Asymmetric scoring)

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Notes

- The code uses TensorFlow 2.x for model implementation
- Genetic algorithm requires multiple model training iterations
- Results may vary due to random initialization and stochastic processes
- GPU acceleration recommended for faster training

## Citation

If you use this code in your research, please cite:
```
[Paper title]
DOI: https://doi.org/10.1115/1.4064809
Dataset: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
```

## License

[Add your license here - e.g., MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- [Your Name]

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Last Updated**: February 2026
