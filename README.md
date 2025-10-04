import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
Loading the Data
train = pd.read_csv('/kaggle/input/week-2/hacktest (1).csv')
test = pd.read_csv('/kaggle/input/week-2/hacktrain (1).csv')

# Drop unnecessary column
train.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
train.head()
Identifying NDVI Columns
ndvi_columns = [col for col in train.columns if '_N' in col]
print(f"NDVI columns: {ndvi_columns[:5]} ... total: {len(ndvi_columns)}")
Handling Missing Values
train[ndvi_columns] = train[ndvi_columns].bfill(axis=1).ffill(axis=1)
test[ndvi_columns] = test[ndvi_columns].bfill(axis=1).ffill(axis=1)
train[ndvi_columns].isnull().sum().sum()
5.Feature Engineering: NDVI Statistics

def add_stats(df):
    df['ndvi_mean'] = df[ndvi_columns].mean(axis=1)
    df['ndvi_std'] = df[ndvi_columns].std(axis=1)
    df['ndvi_max'] = df[ndvi_columns].max(axis=1)
    df['ndvi_min'] = df[ndvi_columns].min(axis=1)
    df['ndvi_range'] = df['ndvi_max'] - df['ndvi_min']
    df['ndvi_median'] = df[ndvi_columns].median(axis=1)
    df['ndvi_skew'] = df[ndvi_columns].skew(axis=1)
    return df

train = add_stats(train)
test = add_stats(test)
train[['ndvi_mean', 'ndvi_std', 'ndvi_max', 'ndvi_min', 'ndvi_range', 'ndvi_median', 'ndvi_skew']].head()
6. Selecting Features

feature_cols = ndvi_columns + [
    'ndvi_mean', 'ndvi_std', 'ndvi_max',
    'ndvi_min', 'ndvi_range', 'ndvi_median', 'ndvi_skew'
]

X = train[feature_cols]
X_test = test[feature_cols]
7. Encoding the Target

le = LabelEncoder()
y = le.fit_transform(train['class'])
print(f"Classes: {le.classes_}")
8. Scaling Features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
9. Creating Polynomial Feature

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f"Original features: {X.shape[1]}, After poly: {X_poly.shape[1]}")
10. Training with Stratified K-Fold

print(f"X_poly shape: {X_poly.shape}")
print(f"y shape: {y.shape}")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros((X_poly.shape[0], len(le.classes_)))
test_preds = np.zeros((X_test_poly.shape[0], len(le.classes_)))

for train_idx, val_idx in skf.split(X_poly, y):
    X_tr, X_val = X_poly[train_idx], X_poly[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    clf = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
    clf.fit(X_tr, y_tr)
    
    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(X_test_poly) / n_splits

from sklearn.metrics import log_loss
print(f"OOF Log Loss: {log_loss(y, oof_preds):.4f}")
11. Pseudo-Labeling: Using Confident Predictions

super_conf_idx = np.where(test_preds.max(axis=1) >= 0.999)[0]
X_pseudo = X_test_poly[super_conf_idx]
y_pseudo = test_preds[super_conf_idx].argmax(axis=1)
print(f"Pseudo-labeled samples: {len(y_pseudo)}")
12. Augmenting the Training Set

X_final = np.vstack([X_poly, X_pseudo])
y_final = np.concatenate([y, y_pseudo])
print(f"Final training set size: {X_final.shape[0]}")
13. Training the Final Model

final_clf = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42)
final_clf.fit(X_final, y_final)
14. Evaluating Model Performance

final_train_preds = final_clf.predict(X_poly)
final_accuracy = accuracy_score(y, final_train_preds)
print(f"Final OOF Accuracy: {final_accuracy:.4f}")
15. Making Predictions for Submission

final_test_preds = final_clf.predict(X_test_poly)
final_test_labels = le.inverse_transform(final_test_preds)
16. Creating the Submission File

submission = pd.DataFrame({
    'ID': test['ID'],
    'class': final_test_labels
})
submission.to_csv('final_submission.csv', index=False)
