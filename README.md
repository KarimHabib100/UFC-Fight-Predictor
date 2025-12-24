# UFC Fight Predictor

A machine learning-based system for predicting UFC fight outcomes using historical fighter statistics and performance metrics.

## Project Purpose

This project aims to build an accurate, explainable, and usable UFC fight outcome predictor. Given two fighters, the model predicts:
- Win probability for each fighter
- Predicted winner with confidence level
- Key factors influencing the prediction

The system is designed to be:
- **Accurate**: Uses advanced ML models with proper validation
- **Explainable**: SHAP-based explanations for transparency
- **Clean and Modular**: Well-structured codebase
- **Usable**: Professional Streamlit GUI

## Features

- Predict fight outcomes between any two UFC fighters
- View win probabilities with confidence intervals
- Understand prediction factors through SHAP explanations
- Explore model performance metrics and calibration
- Time-based train/test splits to prevent data leakage

## Dataset Source

The model is trained on historical UFC fight data including:
- Fighter physical attributes (height, reach, age)
- Career statistics (wins, losses, finishes)
- Performance metrics (strikes, takedowns, submissions)
- Historical fight outcomes

**Note**: Data collection and preprocessing ensure no future information leakage.

## Project Structure

```
ufc_predictor/
│
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned & feature-engineered data
│
├── src/
│   ├── data/
│   │   ├── load.py       # Data loading functions
│   │   ├── clean.py      # Data cleaning functions
│   │   └── features.py   # Feature engineering
│   │
│   ├── models/
│   │   ├── train.py      # Model training
│   │   ├── evaluate.py   # Model evaluation
│   │   └── tune.py       # Hyperparameter optimization
│   │
│   ├── inference/
│   │   └── predict.py    # Prediction pipeline
│   │
│   └── utils/
│       └── helpers.py    # Utility functions
│
├── app/
│   └── streamlit_app.py  # Streamlit GUI
│
├── models/               # Saved trained models
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd UFC-Fight-Predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

### Making Predictions Programmatically

```python
from src.inference.predict import predict_fight

result = predict_fight("Fighter A", "Fighter B")
print(f"Fighter A win probability: {result['fighter_a_win_prob']:.1%}")
print(f"Fighter B win probability: {result['fighter_b_win_prob']:.1%}")
print(f"Predicted winner: {result['predicted_winner']}")
```

## Development Phases

This project is developed in phases:

1. **Phase 1**: Project Setup & Structure *(Current)*
2. **Phase 2**: Data Collection & Cleaning
3. **Phase 3**: Feature Engineering
4. **Phase 4**: Train/Validation/Test Split
5. **Phase 5**: Model Training & Tuning
6. **Phase 6**: Model Evaluation
7. **Phase 7**: Explainability (SHAP)
8. **Phase 8**: Inference Pipeline
9. **Phase 9**: Streamlit GUI
10. **Phase 10**: Final Polish & Documentation

## Expected Outputs

- Trained ML model with tuned hyperparameters
- Win probability predictions for any fighter matchup
- SHAP-based feature explanations
- Calibrated probability estimates
- Interactive Streamlit application

## Tech Stack

- **Python 3.10+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Optimization**: optuna
- **Explainability**: shap
- **Visualization**: matplotlib, seaborn
- **Web Interface**: streamlit
- **Model Persistence**: joblib

## Limitations

- Predictions are probabilistic and not guaranteed
- Model accuracy depends on data quality and recency
- Cannot account for:
  - Fight-night factors (weight cuts, injuries)
  - Mental/psychological factors
  - Training camp quality
  - Rule changes or venue factors
- Past performance doesn't guarantee future results

## License

This project is for educational purposes only.

## Disclaimer

This tool is for educational and entertainment purposes only. It should not be used for gambling or betting decisions. Fight outcomes are inherently unpredictable, and the model provides probability estimates, not certainties.
