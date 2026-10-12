import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score

# =========================================================
# Load
# =========================================================
df = pd.read_csv("course_lead_scoring_2026.csv")

target = 'converted'

# =========================================================
# Data preparation
# =========================================================
cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
num_cols = df.select_dtypes(include=['number']).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

# Fill missing values
df[cat_cols] = df[cat_cols].fillna('NA')
df[num_cols] = df[num_cols].fillna(0.0)

print("Missing values remaining:", df.isnull().sum().sum())

# =========================================================
# Q1 - Mode of 'industry'
# =========================================================
print("\n" + "=" * 55)
print("Question 1: Mode of 'industry'")
print("=" * 55)
print(df['industry'].value_counts())

# =========================================================
# Q2 - Correlation matrix
# =========================================================
print("\n" + "=" * 55)
print("Question 2: Correlation matrix (numerical features)")
print("=" * 55)

pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count'),
]
for a, b in pairs:
    c = df[a].corr(df[b])
    print(f"{a:28s} vs {b:28s} corr = {c:.4f}")

# =========================================================
# Split (exact calls, random_state=42)
# =========================================================
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42
)

df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)

y_train = df_train[target].values
y_val   = df_val[target].values
y_test  = df_test[target].values

del df_train[target]
del df_val[target]
del df_test[target]

print(f"\nSizes -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

# =========================================================
# Q3 - Mutual information (training set only)
# =========================================================
print("\n" + "=" * 55)
print("Question 3: Mutual information score")
print("=" * 55)

for col in ['industry', 'location', 'lead_source', 'employment_status']:
    mi = round(mutual_info_score(df_train[col], y_train), 2)
    print(f"{col:20s} MI = {mi}")

# =========================================================
# Helper for logistic regression
# =========================================================
def prepare_X(df, dv=None):
    dicts = df.to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

X_train, dv = prepare_X(df_train)
X_val, _    = prepare_X(df_val, dv=dv)

# =========================================================
# Q4 - Logistic regression accuracy
# =========================================================
print("\n" + "=" * 55)
print("Question 4: Logistic regression accuracy")
print("=" * 55)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
base_acc = accuracy_score(y_val, y_pred)
print(f"Validation accuracy = {base_acc:.2f}  (raw: {base_acc:.6f})")

# =========================================================
# Q5 - Feature elimination
# =========================================================
print("\n" + "=" * 55)
print("Question 5: Least useful feature")
print("=" * 55)

features = ['lead_source', 'number_of_courses_viewed', 'interaction_count']
for feat in features:
    dtr = df_train.drop(columns=[feat])
    dva = df_val.drop(columns=[feat])
    Xtr, dv_ = prepare_X(dtr)
    Xva, _   = prepare_X(dva, dv=dv_)
    m = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    m.fit(Xtr, y_train)
    acc = accuracy_score(y_val, m.predict(Xva))
    diff = base_acc - acc
    print(f"Without {feat:28s} acc = {acc:.4f}  diff = {diff:+.4f}")

# =========================================================
# Q6 - Regularization
# =========================================================
print("\n" + "=" * 55)
print("Question 6: Regularized logistic regression")
print("=" * 55)

for C in [0.000001, 0.00001, 0.0001, 0.001]:
    m = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_val, m.predict(X_val))
    print(f"C = {C:<10} accuracy = {acc:.3f}")
