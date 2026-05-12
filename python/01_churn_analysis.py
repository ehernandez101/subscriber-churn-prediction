import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/subscriber_churn_dataset.csv')

# KPI calculations
total_users = len(df)
churn_rate = df['churned'].mean() * 100
avg_watch = df['monthly_watch_minutes'].mean()

print(f"Total Users: {total_users}")
print(f"Churn Rate: {churn_rate:.2f}%")
print(f"Average Watch Minutes: {avg_watch:.0f}")

# -----------------------------
# Churn by Subscription Plan
# -----------------------------

plan_churn = (
    df.groupby('subscription_type')['churned']
    .mean()
    .sort_values()
)

plt.figure(figsize=(8,5))
plan_churn.plot(kind='bar')

plt.title('Churn Rate by Subscription Plan')
plt.ylabel('Churn Rate')
plt.xlabel('Subscription Plan')

plt.tight_layout()

plt.savefig('visuals/churn_by_plan.png')

# -----------------------------
# Device Usage
# -----------------------------

device_counts = df['device_type'].value_counts()

plt.figure(figsize=(8,5))
device_counts.plot(kind='bar')

plt.title('Sessions by Device Type')
plt.ylabel('Users')
plt.xlabel('Device')

plt.tight_layout()

plt.savefig('visuals/device_usage.png')

# -----------------------------
# Watch Minutes vs Churn
# -----------------------------

plt.figure(figsize=(8,5))

plt.scatter(
    df['monthly_watch_minutes'],
    df['days_since_last_login'],
    alpha=0.4
)

plt.title('Watch Minutes vs Inactivity')
plt.xlabel('Monthly Watch Minutes')
plt.ylabel('Days Since Last Login')

plt.tight_layout()

plt.savefig('visuals/watch_vs_inactivity.png')

print("Charts created successfully.")
