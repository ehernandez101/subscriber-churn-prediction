import pandas as pd 
import numpy as np
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

n = 10000

plans = ["Basic", "Standard", "Premium"]
devices = ["Mobile", "TV", "Tablet", "Desktop"]
regions = ["North America", "Europe", "South America", "Asia"]

df = pd.DataFrame({
    "user_id": range(1, n + 1),
    "subscription_plan": np.random.choice(plans, n, p=[0.42, 0.38, 0.20]),
    "device": np.random.choice(devices, n, p=[0.45, 0.30, 0.10, 0.15]),
    "region": np.random.choice(regions, n, p=[0.45, 0.25, 0.15, 0.15]),
    "watch_minutes": np.random.normal(120, 40, n).clip(5),
    "login_frequency": np.random.normal(18, 7, n).clip(1),
    "days_inactive": np.random.normal(10, 12, n).clip(0),
    "support_tickets": np.random.poisson(1.2, n),
    "tenure_months": np.random.randint(1, 60, n)
})

risk_score = (
    (df["days_inactive"] * 0.35)
    - (df["watch_minutes"] * 0.015)
    - (df["login_frequency"] * 0.2)
    + (df["support_tickets"] * 1.5)
)

df["churn_probability"] = (1 / (1 + np.exp(-risk_score / 10))).round(3)
df["churned"] = np.where(df["churn_probability"] > 0.5, 1, 0)

df["risk_segment"] = pd.cut(
    df["churn_probability"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low", "Medium", "High"]
)

output = DATA / "subscriber_churn_dataset.csv"
df.to_csv(output, index=False)

print(f"✅ Dataset created: {output}")
print(df.head())
