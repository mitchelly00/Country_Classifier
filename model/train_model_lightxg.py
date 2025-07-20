import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Assume you have your dataframes: train_df, val_df, test_df
train_df = pd.read_pickle("./data_splits/train2.pkl")
test_df = pd.read_pickle("./data_splits/test2.pkl")
# And your features and label columns:
feature_cols = "Combined"  # list your handcrafted + embedding feature columns here
label_col = 'Dance'

# Prepare data
X_train = np.vstack(train_df[feature_cols].values)
y_train = train_df[label_col].values


X_test = np.vstack(test_df[feature_cols].values)
y_test = test_df[label_col].values

# Encode labels to integers for LightGBM
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train_enc)

# Parameters (you can tune these)
params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1,
}

# Train with early stopping
bst = lgb.train(
    params,
    train_data,
    num_boost_round=100,
)

# Predict on test set
y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred = y_pred_proba.argmax(axis=1)

# Print classification report
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
