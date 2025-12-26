# UFC Fight Predictor

A machine learning-based system for predicting UFC fight outcomes using historical fighter statistics and performance metrics.

## Project Purpose

This project aims to build an accurate, explainable, and usable UFC fight outcome predictor. Given two fighters, the model predicts:
- Win probability for each fighter
- Predicted winner with confidence level
- Key factors influencing the prediction (SHAP explanations)

The system is designed to be:
- **Accurate**: 67% test accuracy using LightGBM with Optuna optimization
- **Explainable**: SHAP-based explanations for transparency
- **Clean and Modular**: Well-structured codebase with 108 engineered features
- **Usable**: Professional Streamlit GUI

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Features

- **Fight Prediction**: Predict outcomes between any two UFC fighters
- **Fighter Search**: Search and view statistics for 2,100+ fighters
- **Fight Card Predictions**: Predict entire event cards at once
- **SHAP Explanations**: Understand why predictions are made
- **Model Dashboard**: View model performance and methodology

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 67.4% |
| ROC-AUC | 0.724 |
| Fighters in Database | 2,108 |
| Features Used | 108 |

## Dataset

The model is trained on historical UFC fight data (2010-2024):
- **6,541 fights** with complete statistics
- **Chronological split** to prevent data leakage
- Train: 70% | Validation: 15% | Test: 15%

## Project Structure

```
UFC-Fight-Predictor/
├── app.py                    # Streamlit GUI (run this!)
├── data/
│   ├── raw/                  # Original UFC dataset
│   └── processed/            # Cleaned & feature-engineered data
├── src/
│   ├── data/
│   │   ├── load.py          # Data loading
│   │   ├── clean.py         # Data cleaning
│   │   └── features.py      # Feature engineering
│   ├── models/
│   │   ├── train.py         # Model training
│   │   ├── evaluate.py      # Evaluation metrics
│   │   ├── tune.py          # Optuna optimization
│   │   └── explain.py       # SHAP explanations
│   ├── prediction/
│   │   └── predict.py       # Prediction interface
│   └── utils/
│       └── helpers.py       # Utility functions
├── models/                   # Saved trained models
├── plots/                    # Evaluation plots
├── requirements.txt
└── README.md
```

## Usage

### Running the Web App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Making Predictions Programmatically

```python
from src.prediction import FightPredictor, print_prediction

# Load predictor
predictor = FightPredictor('lightgbm')

# Search for fighters
predictor.search_fighter('McGregor')  # ['Conor Mcgregor']

# Make a prediction
result = predictor.predict(
    red_fighter='Jon Jones',
    blue_fighter='Stipe Miocic',
    weight_class='Heavyweight',
    title_bout=True,
    num_rounds=5,
    red_odds=-300,
    blue_odds=250,
    explain=True  # Include SHAP explanation
)

# Print formatted result
print_prediction(result)

# Access programmatically
print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Predicting Fight Cards

```python
fights = [
    {'red_fighter': 'Jon Jones', 'blue_fighter': 'Stipe Miocic',
     'red_odds': -300, 'blue_odds': 250},
    {'red_fighter': 'Islam Makhachev', 'blue_fighter': 'Charles Oliveira',
     'red_odds': -200, 'blue_odds': 170}
]

results = predictor.predict_card(fights)
for r in results:
    print(f"{r['predicted_winner']} wins ({r['confidence']:.1%})")
```

## Top Predictive Features

1. **RedExpectedValue** - Betting value for Red corner
2. **ImpliedProbDiff** - Difference in implied win probabilities
3. **BlueOdds** - Blue corner betting odds
4. **BlueAvgSigStrLanded** - Blue's striking output
5. **OddsDiff** - Difference in betting odds

## Development Phases

- [x] **Phase 1**: Project Setup & Structure
- [x] **Phase 2**: Data Collection & Cleaning
- [x] **Phase 3**: Feature Engineering (108 features)
- [x] **Phase 4**: Chronological Train/Val/Test Split
- [x] **Phase 5**: Model Training (LR, XGBoost, LightGBM)
- [x] **Phase 6**: Model Evaluation & Comparison
- [x] **Phase 7**: SHAP Explainability
- [x] **Phase 8**: Prediction Interface
- [x] **Phase 9**: Streamlit GUI
- [ ] **Phase 10**: Final Polish & Documentation

## Tech Stack

- **Python 3.10+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Optimization**: optuna
- **Explainability**: shap
- **Visualization**: matplotlib, plotly
- **Web Interface**: streamlit
- **Model Persistence**: joblib

## Limitations

- Predictions are probabilistic, not guaranteed
- Cannot account for:
  - Fight-night factors (injuries, weight cuts)
  - Mental/psychological factors
  - Training camp quality
  - Last-minute changes
- Vegas odds already incorporate significant information
- Professional betting syndicates achieve only 53-58% accuracy

## Disclaimer

This tool is for **educational and entertainment purposes only**. It should not be used for gambling or betting decisions. Fight outcomes are inherently unpredictable, and the model provides probability estimates, not certainties.

## License

This project is for educational purposes only.
