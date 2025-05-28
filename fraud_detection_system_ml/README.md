# E-commerce Fraud Detection System

This project implements a machine learning-based fraud detection system for e-commerce transactions. The system is designed to identify potentially fraudulent transactions using various machine learning algorithms.

## Project Structure

```
fraud_detection_system_ml/
├── data/               # Directory for storing datasets
├── models/            # Directory for saved model files
├── src/              # Source code files
│   ├── generate_sample_data.py  # Script to generate sample transaction data
│   ├── train.py      # Script for model training
│   └── predict.py    # Script for making predictions
├── notebooks/        # Jupyter notebooks for analysis and demos
└── docs/             # Documentation files
```

## Features

- Sample data generation for testing and development
- Multiple ML models including XGBoost and LightGBM
- Model evaluation and comparison tools
- Easy-to-use prediction interface
- Comprehensive documentation

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate sample data (optional):
```bash
python src/generate_sample_data.py --output data/sample_data.csv
```

2. Train the model:
```bash
python src/train.py
```

3. Make predictions:
```bash
python src/predict.py --input data/new_transactions.csv
```

## Model Selection

The system uses an ensemble of XGBoost and LightGBM models, which are particularly well-suited for fraud detection because:
- They handle imbalanced data well
- Can capture complex non-linear patterns
- Provide feature importance analysis
- Are computationally efficient
- Have good performance on tabular data

## Data Format

The system expects transaction data with the following features (column names can be different, the system will map them):
- Transaction amount
- Transaction timestamp
- Customer information
- Product information
- Payment method
- Shipping address
- IP address
- Device information

## Contributing

Feel free to submit issues and enhancement requests. 