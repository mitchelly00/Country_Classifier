import pickle
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd

train_df = pd.read_pickle("./data_splits/train.pkl")
val_df = pd.read_pickle("./data_splits/eval.pkl")
test_df = pd.read_pickle("./data_splits/test.pkl")


# Extract features and labels
X_train = train_df["Feature"].tolist()
y_train = train_df["Dance"]

X_val = val_df["Feature"].tolist()
y_val = val_df["Dance"]

X_test = test_df["Feature"].tolist()
y_test = test_df["Dance"]

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    verbosity=1
)

clf.fit(
    X_train, y_train_enc,
    eval_set=[(X_val, y_val_enc)],
    early_stopping_rounds=10,
    verbose=True
)

from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))

import matplotlib.pyplot as plt
xgb.plot_importance(clf, max_num_features=10)
plt.show()

