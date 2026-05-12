import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VISUALS = ROOT / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "subscriber_churn_dataset.csv")

total_subscribers = len(df)
churn_rate = df["churned"].mean()
retention_rate = 1 - churn_rate
avg_watch_minutes = df["watch_minutes"].mean()

print("===== KPI SUMMARY =====")
print(f"Total Subscribers: {total_subscribers:,}")
print(f"Churn Rate: {churn_rate:.2%}")
print(f"Retention Rate: {retention_rate:.2%}")
print(f"Avg Watch Minutes: {avg_watch_minutes:.1f}")

# Churn by Plan
churn_by_plan = df.groupby("subscription_plan")["churned"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(churn_by_plan["subscription_plan"], churn_by_plan["churned"])
plt.title("Churn Rate by Subscription Plan")
plt.xlabel("Subscription Plan")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(VISUALS / "churn_by_plan.png", dpi=150)
plt.close()

# Churn by Device
churn_by_device = df.groupby("device")["churned"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(churn_by_device["device"], churn_by_device["churned"])
plt.title("Churn Rate by Device")
plt.xlabel("Device")
plt.ylabel("Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(VISUALS / "churn_by_device.png", dpi=150)
plt.close()

# Churn by Region
churn_by_region = df.groupby("region")["churned"].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(churn_by_region["region"], churn_by_region["churned"])
plt.title("Churn Rate by Region")
plt.xlabel("Region")
plt.ylabel("Churn Rate")
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(VISUALS / "churn_by_region.png", dpi=150)
plt.close()

# Watch Minutes: Churned vs Retained
watch_by_churn = df.groupby("churned")["watch_minutes"].mean().reset_index()
watch_by_churn["status"] = watch_by_churn["churned"].map({0: "Retained", 1: "Churned"})

plt.figure(figsize=(8, 5))
plt.bar(watch_by_churn["status"], watch_by_churn["watch_minutes"])
plt.title("Avg Watch Minutes: Retained vs Churned")
plt.xlabel("Subscriber Status")
plt.ylabel("Avg Watch Minutes")
plt.tight_layout()
plt.savefig(VISUALS / "watch_minutes_retained_vs_churned.png", dpi=150)
plt.close()

# Risk Segment Count
risk_counts = df["risk_segment"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
risk_counts.columns = ["risk_segment", "subscribers"]

plt.figure(figsize=(8, 5))
plt.bar(risk_counts["risk_segment"], risk_counts["subscribers"])
plt.title("Subscriber Churn Risk Segments")
plt.xlabel("Risk Segment")
plt.ylabel("Subscriber Count")
plt.tight_layout()
plt.savefig(VISUALS / "churn_risk_segments.png", dpi=150)
plt.close()

print("✅ Churn analysis visuals created.")
