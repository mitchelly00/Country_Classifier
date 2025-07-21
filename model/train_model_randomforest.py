from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Load your data
train_df = pd.read_pickle("./data_splits/train2.pkl")
test_df = pd.read_pickle("./data_splits/test2.pkl")

# Combine features
X_train = np.vstack(train_df['Combined'].values)
X_test = np.vstack(test_df['Combined'].values)
y_train = train_df['Dance']
y_test = test_df['Dance']

# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
clf.fit(X_train, y_train_enc)

# Predict + report
y_pred = clf.predict(X_test)
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
