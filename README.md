# Mate-in-One Chess Move Prediction

**SYDE 522 Final Project — University of Waterloo**

This project develops and compares machine-learning models for predicting the correct mate-in-one move from a chess position. Each position is represented using Forsyth–Edwards Notation (FEN), and the models rank the legal candidate moves to identify the move that immediately checkmates the opponent.

## Project Overview

Chess move prediction is challenging because each board position has a structured and variable set of legal moves. This project formulates mate-in-one prediction as a supervised move-ranking problem.

For each chess position, the pipeline:

1. Parses the FEN string into a chessboard state.
2. Generates every legal move for White.
3. Constructs numerical features for each candidate move.
4. Trains a model to score and rank the legal moves.
5. Evaluates whether the labeled mating move appears in the model’s top predictions.

The project compares feature-based traditional machine-learning models with a convolutional neural network that learns directly from board tensors.

## Models

The following models were implemented and evaluated:

- Logistic Regression
- Ridge Classifier
- Linear Support Vector Machine
- RBF-Kernel Support Vector Machine
- Multi-Layer Perceptron
- Convolutional Neural Network

The regression models, SVMs, and MLP use engineered move-level features. The CNN uses a multi-plane `8 × 8` board representation containing the board state and candidate move.

## Feature Engineering

Each legal candidate move is converted into a fixed-length numerical feature vector containing information such as:

- Moving piece type
- Source and destination squares
- Capture indicator
- Promotion indicator
- Check indicator
- Material changes
- Mobility changes
- King-safety information
- Tactical and positional characteristics

For the CNN, the input is represented using 19 board planes:

- 17 planes describing the board state
- 1 plane identifying the move’s source square
- 1 plane identifying the move’s destination square

## Dataset

The project uses the **Mate in One (Chess)** dataset published on Kaggle.

Each record contains:

- A chess position represented as a FEN string
- The labeled mate-in-one move
- White as the side to move

The separate test dataset contains 5,012 held-out positions.

The dataset is not included in this repository. Place the downloaded files in the following structure:

```text
data/
├── train.csv
└── test.csv

## Evaluation

Models are evaluated at the position level. All candidate moves generated from the same chess position remain in the same data split to prevent data leakage.

The main evaluation metrics are:

- **Top-1 accuracy:** The proportion of positions where the model's highest-ranked move matches the labeled mate-in-one move.
- **Top-5 accuracy:** The proportion of positions where the labeled mate-in-one move appears among the model's five highest-ranked legal moves.
- **Centipawn gap:** The difference between the Stockfish evaluation of the labeled move and the model-selected move.

Group-aware validation is used during hyperparameter tuning so that legal moves from the same position cannot appear in both the training and validation sets.

## Results

| Model | Top-1 Accuracy | Top-5 Accuracy |
|---|---:|---:|
| Logistic Regression | 0.4044 | **0.8957** |
| Ridge Classifier | **0.4064** | 0.8887 |
| Linear SVM | 0.3992 | 0.8939 |
| RBF SVM | 0.3065 | 0.8655 |
| MLP | 0.3901 | 0.8901 |
| CNN | 0.3430 | 0.6927 |

The feature-based linear models achieved the strongest overall ranking performance. Logistic regression obtained the highest test top-5 accuracy, while ridge classification achieved the highest top-1 accuracy.

The nonlinear models did not improve top-1 prediction under the current experimental setup. The CNN was trained on a smaller subset of the available positions and did not learn mating patterns as effectively as the models using engineered move features.

## Key Findings

- Engineered move-level features were effective for ranking mate-in-one moves.
- Most models frequently placed the correct move among their top five predictions.
- Selecting the mating move as the top-ranked candidate remained significantly more difficult.
- Increasing model complexity did not automatically improve prediction performance.
- Missing the mate-in-one move generally produced a large Stockfish-evaluated quality penalty.
- Stockfish mate scores must be handled consistently when converting evaluations into centipawn values.

## Repository Structure

```text
mate-in-one-chess-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── docs/
│   └── SYDE522_Mate_in_One_Final_Paper.pdf
├── final_project.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and Stockfish executable may be excluded from GitHub using `.gitignore`.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

A suitable `requirements.txt` file is:

```text
jupyter
matplotlib
numpy
pandas
python-chess
scikit-learn
torch
torchvision
```

## Dataset Setup

Download the Mate in One chess dataset from Kaggle and place the CSV files in the following directory:

```text
data/
├── train.csv
└── test.csv
```

The dataset is not included in this repository.

## Stockfish Setup

Stockfish is required for engine-based move-quality evaluation.

Store the Stockfish executable locally using a structure such as:

```text
stockfish/
└── stockfish.exe
```

Update the corresponding paths in the notebook:

```python
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
STOCKFISH_PATH = "stockfish/stockfish.exe"
```

The Stockfish filename may differ depending on the operating system and downloaded version.

## Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
final_project.ipynb
```

Run the notebook cells sequentially. The notebook includes:

1. Dataset loading
2. FEN parsing
3. Legal move generation
4. Feature construction
5. Position-based data splitting
6. Model training
7. Hyperparameter tuning
8. Test-set evaluation
9. Stockfish move-quality analysis

The RBF SVM, MLP, and CNN experiments may require substantial computation time.

## Limitations

- The CNN was trained on only a subset of the available training positions.
- Exact top-1 mate selection remained difficult despite relatively high top-5 accuracy.
- Some Stockfish centipawn results were affected by inconsistent terminal mate-score handling.
- Local dataset and Stockfish paths must be configured before running the notebook.
- The evaluation assumes that the move provided by the dataset is the intended mate-in-one label.

## Future Work

Potential improvements include:

- Position-aware masked-softmax training
- Listwise move-ranking losses
- Hard-negative mining within each position
- Additional mate-specific tactical features
- Consistent mate-to-centipawn score conversion
- Larger-scale CNN training
- More expressive spatial neural-network architectures
- Direct verification that predicted moves produce checkmate

## Author

**James Xie**  
Department of Systems Design Engineering  
University of Waterloo
