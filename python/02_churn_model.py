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

BG = "#0B1220"
CARD = "#111827"
TEXT = "#F9FAFB"

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
    n_estimators=150,
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
}).sort_values("importance", ascending=True)

plt.figure(figsize=(9, 5))
fig = plt.gcf()
ax = plt.gca()

fig.set_facecolor(BG)
ax.set_facecolor(CARD)

plt.barh(importance["feature"], importance["importance"], color="#38BDF8")

ax.set_title("Churn Model Feature Importance", color=TEXT, fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Importance", color=TEXT)
ax.set_ylabel("Feature", color=TEXT)
ax.tick_params(colors=TEXT)

for spine in ax.spines.values():
    spine.set_color("#334155")

ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.savefig(VISUALS / "feature_importance.png", dpi=180)
plt.close()

print("✅ Churn model completed.")
print("✅ Feature importance chart created.")
