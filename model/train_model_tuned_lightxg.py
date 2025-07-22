import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import boto3

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

# Define best trial parameters
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': num_classes,
    'learning_rate': 0.024721654545255774,
    'num_leaves': 106,
    'max_depth': 8,
    'min_child_samples': 13,
    'subsample': 0.7177646518647348,
    'colsample_bytree': 0.9386036286584165,
    'reg_alpha': 0.5453364531992095,
    'reg_lambda': 3.5926846115843007,
    'verbose': -1,
}

# Create dataset for training (no validation here)
train_data = lgb.Dataset(X_train_val, label=y_train_val)

# Train model on combined train + val data using best boosting round
bst = lgb.train(
    params,
    train_data,
    num_boost_round=376  # ← Best iteration from Optuna trial
    #verbose_eval=10
)

# Predict on test data
y_pred_proba = bst.predict(X_test)
y_pred = y_pred_proba.argmax(axis=1)

# Print classification report on test set
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#Saving model
file_path = "./final_models/lightgbm_model.txt"
bst.save_model(file_path)

s3 = boto3.client('s3')

s3.upload_file(file_path,'ucwdc-country-classifier', "/final_models/lightgbm_model.txt")