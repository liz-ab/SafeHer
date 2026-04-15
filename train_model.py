import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Load dataset ──
df = pd.read_csv("data.csv")

# ── Encode categorical columns ──
categorical_cols = [
    "Street_Light",
    "CCTV",
    "Police_Patrol",
    "Isolation_Level",
    "Time_Period"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# ── Rule-based risk score calculation ──

# Mapping values (based on real-world intuition)
street_light_map = {"Poor": 1.0, "Moderate": 0.5, "Good": 0.0}
cctv_map         = {"No": 1.0, "Yes": 0.0}
patrol_map       = {"Rare": 1.0, "Occasional": 0.5, "Frequent": 0.0}
isolation_map    = {"High": 1.0, "Medium": 0.5, "Low": 0.0}
time_map         = {"Night": 1.0, "Evening": 0.7, "Afternoon": 0.3, "Morning": 0.2}

# Reload original data (non-encoded) for mapping
df_raw = pd.read_csv("data.csv")

# Normalize crime count
crime_min = df_raw["Crime_Count"].min()
crime_max = df_raw["Crime_Count"].max()
crime_norm = (df_raw["Crime_Count"] - crime_min) / (crime_max - crime_min)

# Final risk score formula
risk_score = (
    0.30 * crime_norm +
    0.20 * df_raw["Street_Light"].map(street_light_map) +
    0.15 * df_raw["CCTV"].map(cctv_map) +
    0.20 * df_raw["Police_Patrol"].map(patrol_map) +
    0.10 * df_raw["Isolation_Level"].map(isolation_map) +
    0.05 * df_raw["Time_Period"].map(time_map)
)

# Clip values between 0 and 1
df["Risk_Score"] = risk_score.clip(0, 1)

# ── Preview output ──
print("Sample processed data:")
print(df.head())
