import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_val, y_train, y_val = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier()
model.fit(
    X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True
)
