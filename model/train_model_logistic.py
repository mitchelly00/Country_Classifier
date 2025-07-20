import pandas as pd

train_df = pd.read_pickle('data_splits/train2.pkl')
test_df = pd.read_pickle('data_splits/test2.pkl')

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Convert list embeddings to array
X_train = np.vstack(train_df['Combined'].values)
X_test = np.vstack(test_df['Combined'].values)

# Encode string labels into numbers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['Dance'])
y_test = label_encoder.transform(test_df['Dance'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)

# Evaluate
y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))
