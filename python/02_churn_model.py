import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VISUALS = ROOT / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "subscriber_churn_dataset.csv")

features = [
    "watch_minutes",
    "login_frequency",
    "days_inactive",
    "support_tickets",
    "tenure_months"
]

X = df[features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions))

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(importance["feature"], importance["importance"])
plt.title("Churn Model Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(VISUALS / "feature_importance.png", dpi=150)
plt.close()

print("✅ Churn model completed.")
print("✅ Feature importance chart created.")
