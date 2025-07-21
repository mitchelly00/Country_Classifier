import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load data
df_train = pd.read_pickle("./data_splits/train.pkl")
df_val = pd.read_pickle("./data_splits/val.pkl")
df_test = pd.read_pickle("./data_splits/test.pkl")

# Combine train + val data for final training
df_train_val = pd.concat([df_train, df_val], ignore_index=True)

# Label encoding
label_encoder = LabelEncoder()
label_encoder.fit(df_train_val["Dance"].tolist() + df_test["Dance"].tolist())

num_classes = len(label_encoder.classes_)

# Prepare labels
y_train_val = label_encoder.transform(df_train_val["Dance"].values)
y_test = label_encoder.transform(df_test["Dance"].values)

# Prepare features - stacking "Combined" vectors
X_train_val = np.stack(df_train_val["Combined"].values).astype(np.float32)
X_test = np.stack(df_test["Combined"].values).astype(np.float32)

# Define parameters from best trial
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': num_classes,
    'learning_rate': 0.2142407590040286,
    'num_leaves': 124,
    'max_depth': 14,
    'min_child_samples': 15,
    'subsample': 0.997518914220819,
    'colsample_bytree': 0.7145002583288589,
    'reg_alpha': 0.51397955058898,
    'reg_lambda': 2.8529458079534944,
    'verbose': -1,
}

# Create dataset for training (no validation here)
train_data = lgb.Dataset(X_train_val, label=y_train_val)

# Train model on combined train + val data
bst = lgb.train(
    params,
    train_data,
    num_boost_round=1000,  # Or set a number you want
    verbose_eval=10
)

# Predict on test data
y_pred_proba = bst.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

# Print classification report on test set
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
