import pickle
import xgboost as xgb; print(xgb.__version__)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pandas as pd

train_df = pd.read_pickle("./data_splits/train2.pkl")
test_df = pd.read_pickle("./data_splits/test2.pkl")


# Extract features and labels
X_train = train_df["Combined"].tolist()
y_train = train_df["Dance"]


X_test = test_df['Combined'].tolist()
y_test = test_df["Dance"]

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    #early_stopping_rounds=10,
    verbosity=1
)

print(type(clf))

clf.fit(
    X_train, y_train_enc,
    verbose=True
)

from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))



