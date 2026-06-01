# Train Heart Disease prediction model (India-context dataset simulation)
# Features based on common clinical indicators used in Indian cardiology studies
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

np.random.seed(42)
n = 1000

# Simulate India-context heart disease dataset
# Features: age, sex(1=M,0=F), chest_pain(0-3), resting_bp, cholesterol,
#           fasting_blood_sugar(>120=1), ecg(0-2), max_hr, exercise_angina(0/1),
#           st_depression, family_history(0/1), smoking(0/1), diabetes(0/1)

age           = np.random.randint(25, 80, n)
sex           = np.random.randint(0, 2, n)
chest_pain    = np.random.randint(0, 4, n)
resting_bp    = np.random.randint(90, 200, n)
cholesterol   = np.random.randint(150, 400, n)
fbs           = np.random.randint(0, 2, n)
ecg           = np.random.randint(0, 3, n)
max_hr        = np.random.randint(60, 202, n)
exercise_ang  = np.random.randint(0, 2, n)
st_depression = np.round(np.random.uniform(0, 6.2, n), 1)
family_hist   = np.random.randint(0, 2, n)
smoking       = np.random.randint(0, 2, n)
diabetes      = np.random.randint(0, 2, n)

# Risk score formula (India-relevant weightings)
risk = (
    (age > 45).astype(int) * 2 +
    sex * 1 +
    (chest_pain >= 2).astype(int) * 2 +
    (resting_bp > 140).astype(int) * 1.5 +
    (cholesterol > 240).astype(int) * 1.5 +
    fbs * 1 +
    (max_hr < 120).astype(int) * 1 +
    exercise_ang * 2 +
    (st_depression > 2).astype(int) * 2 +
    family_hist * 1.5 +
    smoking * 1.5 +
    diabetes * 2
)
target = (risk + np.random.normal(0, 1.5, n) > 8).astype(int)

X = np.column_stack([age, sex, chest_pain, resting_bp, cholesterol,
                     fbs, ecg, max_hr, exercise_ang, st_depression,
                     family_hist, smoking, diabetes])
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=500, random_state=42))
])
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')
acc = model.score(X_test, y_test)
print(f"✅ Heart Disease Model saved! Accuracy: {acc:.2%}")
print(f"   Positive cases in dataset: {target.sum()} / {n}")
