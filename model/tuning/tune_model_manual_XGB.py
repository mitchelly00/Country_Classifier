import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import LabelEncoder



# Load preprocessed data
df_train = pd.read_pickle("./data_splits/train.pkl")
df_val = pd.read_pickle("./data_splits/val.pkl")
df_test = pd.read_pickle("./data_splits/test.pkl")

# Initialize and fit on all labels to ensure consistent encoding
label_encoder = LabelEncoder()
label_encoder.fit(df_train["Dance"].tolist() + df_val["Dance"].tolist() + df_test["Dance"].tolist())

# Transform the labels
y_train = label_encoder.transform(df_train["Dance"].values)
y_val = label_encoder.transform(df_val["Dance"].values)
y_test = label_encoder.transform(df_test["Dance"].values)


# Use vector column "Combined" for features
X_train = np.stack(df_train["Combined"].values).astype(np.float32)

X_val = np.stack(df_val["Combined"].values).astype(np.float32)

X_test = np.stack(df_test["Combined"].values).astype(np.float32)


# Define hyperparameter space
param_dist = {
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10],
}

n_iter = 30
param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

best_score = 0
best_params = None
best_model = None

# Manual hyperparameter tuning loop
for i, params in enumerate(param_list):
    print(f"Trial {i+1}/{n_iter}: {params}")
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        **params
    )
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    score = accuracy_score(y_val, val_preds)
    print(f"→ Validation Accuracy: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_params = params
        best_model = model

# ✅ Retrain final model on full train + val set with best params
X_train_val = np.stack(df_train_val["Combined"].values).astype(np.float32)
y_train_val = label_encoder.transform(df_train_val["Dance"].values)

final_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    **best_params
)
final_model.fit(X_train_val, y_train_val)

# 🧪 Final test evaluation
test_preds = final_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print("\n✅ Best hyperparameters found:")
print(best_params)
print(f"Best validation accuracy during tuning: {best_score:.4f}")
print(f"🧪 Final test accuracy: {test_acc:.4f}")