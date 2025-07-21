import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def objective(trial):
    param = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': num_classes,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'verbose': -1,
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    gbm = lgb.train(
        param, 
        lgb_train, 
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    pred_labels = preds.argmax(axis=1)
    
    accuracy = accuracy_score(y_val, pred_labels)
    return accuracy

# Load preprocessed data
df_train = pd.read_pickle("./data_splits/train.pkl")
df_val = pd.read_pickle("./data_splits/val.pkl")
df_test = pd.read_pickle("./data_splits/test.pkl")

# Initialize and fit on all labels to ensure consistent encoding
label_encoder = LabelEncoder()
label_encoder.fit(df_train["Dance"].tolist() + df_val["Dance"].tolist() + df_test["Dance"].tolist())

# Define number of classes for LightGBM
num_classes = len(label_encoder.classes_)

# Transform the labels
y_train = label_encoder.transform(df_train["Dance"].values)
y_val = label_encoder.transform(df_val["Dance"].values)


# Use vector column "Combined" for features
X_train = np.stack(df_train["Combined"].values).astype(np.float32)
X_val = np.stack(df_val["Combined"].values).astype(np.float32)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('Best trial:', study.best_trial.params)
print('Best accuracy:', study.best_trial.value)
