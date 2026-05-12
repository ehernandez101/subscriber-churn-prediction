import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VISUALS = ROOT / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "subscriber_churn_dataset.csv")

BG = "#0B1220"
CARD = "#111827"
TEXT = "#F9FAFB"

def style_chart(title, xlabel, ylabel):
    fig = plt.gcf()
    ax = plt.gca()

    fig.set_facecolor(BG)
    ax.set_facecolor(CARD)

    ax.set_title(title, color=TEXT, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, color=TEXT, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT, fontsize=11)

    ax.tick_params(colors=TEXT)

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(axis="y", alpha=0.25)

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

plt.figure(figsize=(9, 5))
plt.bar(churn_by_plan["subscription_plan"], churn_by_plan["churned"], color="#38BDF8")
style_chart("Churn Rate by Subscription Plan", "Subscription Plan", "Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(VISUALS / "churn_by_plan.png", dpi=180)
plt.close()

# Churn by Device
churn_by_device = df.groupby("device")["churned"].mean().reset_index()

plt.figure(figsize=(9, 5))
plt.bar(churn_by_device["device"], churn_by_device["churned"], color="#A78BFA")
style_chart("Churn Rate by Device", "Device", "Churn Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(VISUALS / "churn_by_device.png", dpi=180)
plt.close()

# Churn by Region
churn_by_region = df.groupby("region")["churned"].mean().reset_index()

plt.figure(figsize=(9, 5))
plt.bar(churn_by_region["region"], churn_by_region["churned"], color="#22C55E")
style_chart("Churn Rate by Region", "Region", "Churn Rate")
plt.xticks(rotation=25)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(VISUALS / "churn_by_region.png", dpi=180)
plt.close()

# Watch Minutes: Retained vs Churned
watch_by_churn = df.groupby("churned")["watch_minutes"].mean().reset_index()
watch_by_churn["status"] = watch_by_churn["churned"].map({0: "Retained", 1: "Churned"})

plt.figure(figsize=(9, 5))
plt.bar(watch_by_churn["status"], watch_by_churn["watch_minutes"], color=["#22C55E", "#EF4444"])
style_chart("Avg Watch Minutes: Retained vs Churned", "Subscriber Status", "Avg Watch Minutes")
plt.tight_layout()
plt.savefig(VISUALS / "watch_minutes_retained_vs_churned.png", dpi=180)
plt.close()

# Risk Segments
risk_counts = df["risk_segment"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
risk_counts.columns = ["risk_segment", "subscribers"]

plt.figure(figsize=(9, 5))
plt.bar(risk_counts["risk_segment"], risk_counts["subscribers"], color=["#22C55E", "#F59E0B", "#EF4444"])
style_chart("Subscriber Churn Risk Segments", "Risk Segment", "Subscriber Count")
plt.tight_layout()
plt.savefig(VISUALS / "churn_risk_segments.png", dpi=180)
plt.close()

print("✅ Upgraded churn analysis visuals created.")
