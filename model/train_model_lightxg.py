import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Assume you have your dataframes: train_df, val_df, test_df
# And your features and label columns:
feature_cols = [...]  # list your handcrafted + embedding feature columns here
label_col = 'Dance'

# Prepare data
X_train = train_df[feature_cols].values
y_train = train_df[label_col].values

X_val = val_df[feature_cols].values
y_val = val_df[label_col].values

X_test = test_df[feature_cols].values
y_test = test_df[label_col].values

# Encode labels to integers for LightGBM
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train_enc)
val_data = lgb.Dataset(X_val, label=y_val_enc, reference=train_data)

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
    num_boost_round=200,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    early_stopping_rounds=10,
    verbose_eval=10,
)

# Predict on test set
y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred = y_pred_proba.argmax(axis=1)

# Print classification report
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
