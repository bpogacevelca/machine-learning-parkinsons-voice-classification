import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("all_dataset.csv")

X = df.drop(columns=["status", "File_Name"])
y = df["status"]

# Models to compare
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}

# 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n5-FOLD CROSS-VALIDATION RESULTS\n")

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(name)
    print("Fold scores:", scores)
    print("Mean accuracy:", round(scores.mean() * 100, 2), "%")
    print("Standard deviation:", round(scores.std() * 100, 2), "%")
    print()
    import os
    import joblib
    from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

final_model.fit(X, y)

joblib.dump(final_model, "models/random_forest_model.joblib")

print("Saved final Random Forest model.")