# Credit Card Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using ensemble methods and advanced preprocessing techniques.

## 📊 Project Overview

This project implements a robust credit card fraud detection system using multiple machine learning algorithms combined in a voting classifier. The system achieves high performance with an F1-score of **0.817** and PR-AUC of **0.805** on the validation set.

## 🎯 Key Features

- **Ensemble Learning**: Combines Logistic Regression, Random Forest, and Neural Network (MLP) classifiers
- **Advanced Preprocessing**: Implements SMOTE and undersampling techniques for handling imbalanced data
- **Optimal Threshold Tuning**: Automatically finds the best classification threshold for maximum F1-score
- **Comprehensive Evaluation**: Uses F1-score and PR-AUC metrics suitable for imbalanced datasets
- **Modular Architecture**: Clean, maintainable code structure with separate modules for data processing, training, and evaluation

## 📁 Project Structure

```
Credit_Card_Fraud_Detection/
├── data/                          # Dataset files
│   ├── train.csv                  # Training data
│   ├── val.csv                    # Validation data
│   ├── test.csv                   # Test data
│   └── trainval.csv               # Combined train+validation data
├── src/                           # Source code
│   ├── credit_fraud_train.py      # Main training script
│   ├── credit_fraud_utils_data.py # Data preprocessing utilities
│   ├── credit_fraud_utils_eval.py # Evaluation utilities
│   └── __init__.py
├── models/                        # Trained models
│   └── fraud_model.pkl            # Saved ensemble model
├── results/                       # Results and metrics
│   ├── evaluation_results.json    # Model performance metrics
│   └── best_threshold.txt         # Optimal classification threshold
├── notebooks/                     # Jupyter notebooks
│   └── Credit_Card_eda.ipynb      # Exploratory data analysis
├── reports/                       # Generated reports
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Credit_Card_Fraud_Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script:**
   ```bash
   python src/credit_fraud_train.py
   ```

### Training with Custom Parameters

```bash
python src/credit_fraud_train.py \
    --train_path data/train.csv \
    --val_path data/val.csv \
    --model_path models/fraud_model.pkl \
    --results_path results/evaluation_results.json \
    --threshold_path results/best_threshold.txt \
    --sampling smote \
    --n_estimators 100 \
    --max_depth 10
```

## 🔧 Model Architecture

### Ensemble Classifier
The system uses a **Voting Classifier** that combines three base models:

1. **Logistic Regression**
   - Balanced class weights
   - L1 regularization (liblinear solver)
   - Maximum 1000 iterations

2. **Random Forest**
   - 100 estimators (configurable)
   - Maximum depth of 10 (configurable)
   - Balanced class weights

3. **Multi-Layer Perceptron (MLP)**
   - Hidden layers: (16, 32)
   - Adam optimizer
   - Early stopping with validation fraction 0.1

### Data Preprocessing
- **Feature Scaling**: StandardScaler for all features
- **Sampling Techniques**: 
  - SMOTE for oversampling
  - Random undersampling
  - No sampling option available
- **Validation**: No sampling applied to validation data

## 📈 Performance Results

### Model Performance Metrics
- **F1-Score**: 0.817 (with optimal threshold)
- **PR-AUC**: 0.805
- **Optimal Threshold**: 0.46

### Threshold Optimization
The system automatically finds the optimal classification threshold by:
- Testing thresholds from 0.0 to 1.0 in 0.01 increments
- Selecting the threshold that maximizes F1-score
- Saving the optimal threshold for inference

## 🛠️ Usage

### Training
```python
from src.credit_fraud_train import train_model
import argparse

# Set up arguments
args = argparse.Namespace(
    train_path='data/train.csv',
    val_path='data/val.csv',
    model_path='models/fraud_model.pkl',
    results_path='results/evaluation_results.json',
    threshold_path='results/best_threshold.txt',
    sampling='smote',
    n_estimators=100,
    max_depth=10
)

# Train the model
train_model(args)
```

### Inference
```python
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('models/fraud_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
threshold = model_data['threshold']

# Preprocess new data
def predict_fraud(new_data):
    # Scale features
    X_scaled = scaler.transform(new_data)
    
    # Get probabilities
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Apply optimal threshold
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities
```

## 📊 Data Analysis

The project includes a comprehensive Jupyter notebook (`notebooks/Credit_Card_eda.ipynb`) for:
- Exploratory data analysis
- Feature distribution analysis
- Class imbalance visualization
- Correlation analysis
- Data quality assessment

## 🔍 Configuration Options

### Sampling Strategies
- `none`: No sampling (original data)
- `smote`: SMOTE oversampling
- `undersampling`: Random undersampling

### Model Parameters
- `n_estimators`: Number of trees in Random Forest (default: 100)
- `max_depth`: Maximum depth of trees (default: 10)
- `sampling`: Sampling strategy for training data

## 📝 Output Files

### Model Files
- `models/fraud_model.pkl`: Serialized ensemble model with scaler and threshold

### Results Files
- `results/evaluation_results.json`: Model performance metrics
- `results/best_threshold.txt`: Optimal classification threshold

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Credit Card Fraud Detection dataset
- Libraries: scikit-learn, pandas, numpy, imbalanced-learn
- Techniques: SMOTE, ensemble methods, threshold optimization

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This project is designed for educational and research purposes. Always ensure compliance with relevant regulations when implementing fraud detection systems in production environments.
