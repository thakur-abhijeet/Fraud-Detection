# Comprehensive Guide: E-commerce Fraud Detection System

This guide provides detailed instructions for setting up, training, and using the fraud detection system.

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## System Overview

The fraud detection system uses an ensemble of XGBoost and LightGBM models to identify potentially fraudulent transactions. The system is designed to be flexible and can handle various data formats.

### Key Components:
- **Data Generation Module**: Creates sample transaction data for testing
- **Training Module**: Implements model training and evaluation
- **Prediction Module**: Makes predictions on new transactions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fraud_detection_system_ml
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Sample Data Generation
The system includes a script to generate sample transaction data for testing:

```bash
python src/generate_sample_data.py --output data/sample_data.csv
```

### Required Data Format
The system expects transaction data with the following features (column names can be different):

| Feature | Example Column Names | Description |
|---------|---------------------|-------------|
| Amount | amount, transaction_amount, price | Transaction value |
| Timestamp | timestamp, date, time | Transaction time |
| Customer ID | customer_id, user_id | Unique customer identifier |
| Product ID | product_id, item_id | Unique product identifier |
| Payment Method | payment_method, payment_type | Payment method used |
| Shipping Address | shipping_address, delivery_address | Delivery location |
| IP Address | ip_address, ip | Customer's IP address |
| Device ID | device_id, device | Customer's device identifier |

## Model Training

### Training Process:
1. Run the training script:
```bash
python src/train.py --input data/processed_data.csv --output models/
```

The training process:
- Splits data into training and validation sets
- Trains XGBoost and LightGBM models
- Evaluates model performance
- Saves trained models

### Model Configuration:
The current configuration uses the following parameters:

#### XGBoost:
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

#### LightGBM:
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

## Making Predictions

### Prediction Process:
1. Prepare new transaction data in CSV format
2. Run the prediction script:
```bash
python src/predict.py --input data/new_transactions.csv --output predictions.csv --model_dir models/
```

### Output Format:
The prediction script generates a CSV file with the following additional columns:
- `predicted_fraud`: Binary prediction (0: legitimate, 1: fraudulent)
- `fraud_probability`: Probability of fraud (0 to 1)

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**
   - Error: "ModuleNotFoundError: No module named 'xgboost'"
   - Solution: Run `pip install -r requirements.txt`

2. **Data Format Issues**
   - Error: "KeyError: 'is_fraud'"
   - Solution: Ensure your training data includes the 'is_fraud' column

3. **Memory Issues**
   - Error: "MemoryError"
   - Solution: Reduce batch size or use a machine with more RAM

4. **Model Loading Issues**
   - Error: "FileNotFoundError: [Errno 2] No such file or directory"
   - Solution: Ensure models are saved before making predictions

## Best Practices

### Data Quality:
1. Ensure data completeness
2. Handle outliers appropriately
3. Maintain consistent data formats
4. Regular data validation

### Model Maintenance:
1. Regular model retraining
2. Monitor model performance
3. Update feature engineering as needed
4. Keep track of model versions

### Security:
1. Secure model storage
2. Validate input data
3. Monitor prediction patterns
4. Regular security audits

### Performance Optimization:
1. Use appropriate batch sizes
2. Optimize feature engineering
3. Regular model tuning
4. Monitor system resources

## Additional Resources

- Documentation:
  - `README.md`: Project overview
  - `docs/`: Additional documentation

## Support

For issues and feature requests, please:
1. Check the troubleshooting guide
2. Review existing documentation
3. Submit an issue on the repository
4. Contact the development team 
