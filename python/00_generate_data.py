import pandas as pd
import numpy as np

np.random.seed(42)

num_users = 5000

subscription_types = ['basic', 'standard', 'premium']
devices = ['mobile', 'tv', 'web', 'tablet']
regions = ['US', 'Canada', 'UK', 'LATAM']

data = {
    'user_id': range(1, num_users + 1),

    'subscription_type': np.random.choice(
        subscription_types,
        num_users,
        p=[0.4, 0.35, 0.25]
    ),

    'device_type': np.random.choice(
        devices,
        num_users,
        p=[0.45, 0.25, 0.20, 0.10]
    ),

    'region': np.random.choice(
        regions,
        num_users
    ),

    'monthly_watch_minutes': np.random.normal(
        2200,
        700,
        num_users
    ).astype(int),

    'login_frequency': np.random.randint(
        1,
        30,
        num_users
    ),

    'days_since_last_login': np.random.randint(
        0,
        45,
        num_users
    ),

    'tenure_months': np.random.randint(
        1,
        60,
        num_users
    ),

    'support_tickets': np.random.poisson(
        1.2,
        num_users
    )
}

df = pd.DataFrame(data)

df['monthly_watch_minutes'] = df['monthly_watch_minutes'].clip(lower=100)

df['churned'] = np.where(
    (
        (df['days_since_last_login'] > 20) &
        (df['monthly_watch_minutes'] < 1500)
    ),
    1,
    0
)

df.to_csv('data/subscriber_churn_dataset.csv', index=False)

print("Dataset created successfully.")
print(df.head())
