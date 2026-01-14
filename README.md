# Inverse Problem Solving for Preference Dynamics

<!-- intro -->
**Goal:** This portfolio project combines a data science exploration and machine learning project for solving the inverse problem for a system of nonlinear differential equations (DE).
While solving differential equations is (more or less) straightforward, its **inverse problem** is far more difficult: We want to determine the original parameters of the differential equation only from its solution.
For these kind of problems, neural networks are well-suited, as they can pick up implicit patterns in the data, and scale well for high-dimensional input and output data.

<!-- preference dynamics -->
I apply my custom made ML experimentation suite to an unsolved research problem: Finding the input parameters for a model of human behavior from its time series alone.
The model describes people's desires for activities, and how much effort they undertake to do them ([Krecik, 2025a](https://doi.org/10.1007/s10614-025-10895-3), [Krecik, 2025b](https://www.ssrn.com/abstract=5303381)).

Solving the inverse problem would be highly impactful: If you give me a time series of your day-to-day behavior, I can predict "your parameters" and use them to accurately forecast your behavior far into the future.

<!-- model -->
I will use **convolutional neural networks** (CNNs) and **physics informed neural networks** (PINNs) (planned) to tackle the problem.

*Why CNNs?* Just like 2d CNNs can find patterns in images, 1d CNNs are well-suited to find patterns in 1d signals, like time series. Convolution kernels can detect patterns anywhere in the time series, making it perfectly suited for repeating activities, as we have here.

*Why PINNs?* Since I know which DE I want to predict, I can directly bake it into the model through auto-differentiation and in the loss function. Through this constraint, the model will not just learn any roughly similar DE, but is strongly guided towards learning *this* DE.

<!-- training -->

## Project overview

The project consists of a feature-rich experimentation suite for model training using pytorch, and a series of notebooks for synthetic data generation, exploration, feature engineering and training.

### Experimentation suite

**Data Generation & Validation:**
- Parallel ODE solver with joblib for efficient batch generation of synthetic data
- Automated stability detection using eigenvalue analysis of system Jacobians

**Data Pipeline:**
- Modular `DataManager` for convenient data loading, preprocessing, and feature engineering
- Extensible through protocol-based IO handlers (JSON, Pickle, Parquet (planned))
- Composable transformer pipeline for data cleaning, feature engineering, and normalization. Currently implemented:
    - Peak detection with statistics extraction
    - Limit cycle detection using cycle splitting and mean/diff analysis
    - Steady-state detection
    - Initial condition extractor
    - Data cleaning transformer
    - Sample-level and group-level normalization with statistics tracking
- Multiple `InputAdapter` and `TargetAdapter`s for flexible model inputs and targets specification
- Automatic caching of preprocessed data for fast iteration
- Deterministic train/val/test splitting with seed control

**Training Infrastructure:**
- Model-agnostic `Trainer` compatible with any PyTorch model implementing `PredictorModel` protocol
- Complete checkpointing system saving:
    - Model and optimizer state
    - Random number generator states (PyTorch, NumPy, Python) for full reproducibility
    - Training configuration and metadata
    - Best and last epoch checkpoints
- Early stopping with configurable patience
- Gradient clipping for training stability
- Automatic device selection (CUDA/MPS/CPU) with graceful fallback
- MLflow integration with automatic metric logging and graceful degradation when unavailable

**Experiment Management:**
- `ExperimentRunner` orchestrating complete experiment lifecycle
- Automated hyperparameter studies using Optuna with parallelized trial execution and study resumption
- Customizable parameter suggestion via subclassing
- Multi-objective optimization support (minimize/maximize multiple metrics)

**SageMaker:**
- Scripts for training and deploying to an inference endpoint of the residual CNN model with SageMaker

**Code Quality & Architecture:**
- Type-safe extensive configuration for all classes using Pydantic schemas
- Follows software engineering best practices with protocol-based abstractions and dependency injection, comprehensive testing, and thorough documentation
- Full reproducibility: seed control for all random operations, checkpoint restoration
- Modern Python dev standards: uv, ruff, mypy, pre-commit hooks


*Why not pytorch lightning?* Of course pytorch lightning is incredibly convenient, but it's not the industry standard. I rather want to demonstrate fluency with vanilla pytorch and show off a well-designed custom implementation with full control over model training and tracking.

### Notebooks

View the notebooks in nbviewer:
- [`notebooks/10_data_generation.ipynb`](https://nbviewer.org/github/markuskrecik/preference-dynamics-learning/blob/main/notebooks/10_data_generation.ipynb): Synthetic data generation
- [`notebooks/20_data_exploration.ipynb`](https://nbviewer.org/github/markuskrecik/preference-dynamics-learning/blob/main/notebooks/20_data_exploration.ipynb): Data exploration, cleaning, and visualization
- [`notebooks/30_feature_engineering.ipynb`](https://nbviewer.org/github/markuskrecik/preference-dynamics-learning/blob/main/notebooks/30_feature_engineering.ipynb): Feature engineering & baseline linear regression model
- [`notebooks/40_training_cnn_n1.ipynb`](https://nbviewer.org/github/markuskrecik/preference-dynamics-learning/blob/main/notebooks/40_training_cnn_n1.ipynb): Naive 1d CNN model training, optimization, and evaluation for n=1 action
- [`notebooks/41_training_cnn_n1_residual.ipynb`](https://nbviewer.org/github/markuskrecik/preference-dynamics-learning/blob/main/notebooks/41_training_cnn_n1_residual.ipynb): Residual 1d CNN model training, optimization, and evaluation for n=1 action
- [`notebooks/workbench/60_sagemaker.ipynb`](https://nbviewer.org/github/markuskrecik/preference-dynamics-learning/blob/main/notebooks/workbench/60_sagemaker.ipynb): SageMaker notebook for training and deploying the model.

### Model Architectures

#### 0. Linear Regression: Baseline
- Linear regression with manual and automated (tsfresh) feature extraction
- Easy to implement and interpret

#### 1. 1d Convolutional Neural Network (CNN)
- Good for local pattern detection in repeating time series
- Uses adaptive pooling for variable-length sequences

#### 2. Physics-Informed Neural Network (PINN)
- Incorporates DE structure into learning
- Better generalization through constraint enforcement


## Quick Start

### Installation

#### 1. Install uv (package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Clone and set up project

```bash
git clone https://github.com/markuskrecik/preference-dynamics-learning.git
cd preference-dynamics-learning

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Install pre-commit hooks (optional)
pre-commit install
```

### Run

Explore the project through Jupyter notebooks:

```bash
uv run jupyter lab
```

1. Generate Data: `notebooks/10_data_generation.ipynb`
2. Explore Data: `notebooks/20_data_exploration.ipynb`
3. Feature Engineering & Linear Regression: `notebooks/30_feature_engineering.ipynb`
4. CNN Model training and evaluation:
    - for n=1 action: `notebooks/40_training_cnn_n1.ipynb`
    - for n=1 action with residual architecture: `notebooks/41_training_cnn_n1_residual.ipynb`
5. SageMaker Training and Deployment: `notebooks/60_sagemaker.ipynb`

#### Experiment Tracking

All experiments are automatically tracked with MLflow:

```bash
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
# Open http://localhost:5000
```

**Tracked artifacts**:
- Hyperparameters
- Training, validation, and test metrics
- Model checkpoints
- Dataset statistics


## Project Structure

```
preference-dynamics-learning/
├── README.md               # This file
├── LICENSE.md              # License
├── pyproject.toml          # Project config, dependencies, tool settings
├── .pre-commit-config.yaml # Pre-commit hooks
├── mlruns.db               # MLflow tracking database (not in git)
├── optuna.db               # Optuna hyperparameter study tracking database (not in git)
│
├── src/                    # Source code
│   └── preference_dynamics/
│       ├── solver/         # ODE solver and data generation
│       ├── data/           # Data pipeline and validation
│       ├── models/         # ML model architectures
│       ├── training/       # Training infrastructure
│       ├── experiments/    # Experiment orchestration
│       └── visualization/  # Plotting utilities
│
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
│
├── notebooks/              # Jupyter notebooks
│   ├── 10_data_generation.ipynb
│   ├── 20_data_exploration.ipynb
│   ├── 30_feature_engineering.ipynb
│   ├── 40_training_cnn_n1.ipynb
│   └── ...
│
├── data/                   # Generated data (not in git)
│   ├── n1/                 # Data for n=1 actions
│   │   ├── raw/
│   │   └── processed/
│   ├── n2/                 # Data for n=2 actions
│   │   └── ...
│   └── n3/                 # Data for n=3 actions
│   │   └── ...
│
├── checkpoints/            # Trainer state checkpoints (not in git)
└── mlruns/                 # MLflow model checkpoints (not in git)
```


## Running Tests

```bash
# All tests
uv run pytest -v

# With coverage
uv run pytest --cov=src/preference_dynamics --cov-report=html

# Skip slow tests
uv run pytest -m "not slow"
```


## Development

### Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check --fix src/ tests/

# Type checking
uv run mypy src/

# Run all quality checks
uv run pre-commit run --all-files
```

### Project Configuration

All tools configured in `pyproject.toml`:
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Testing
- **pre-commit**: Git hooks

## Tech Stack

- **Language**: Python 3.11
- **Package Manager**: uv
- **Numerical Computing**: NumPy, SciPy
- **Deep Learning**: PyTorch
- **ML Utilities**: optuna, scikit-learn, tsfresh
- **Experiment Tracking**: MLflow
- **Cloud Computing**: SageMaker
- **Visualization**: Plotly
- **Data Analysis**: Pandas
- **Testing**: pytest
- **Code Quality**: ruff, mypy, pre-commit

## License

GNU General Public License v3.0 - see [LICENSE.md](LICENSE.md) file for details.

## Links

- **Repository**: https://github.com/markuskrecik/preference-dynamics-learning

## Contact

**Author**: Markus Krecik
