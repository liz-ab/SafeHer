from flask import Flask, render_template, request
import pandas as pd
import pickle
import folium
from folium.plugins import HeatMap
import os

app = Flask(__name__)

# ── Load ML model and encoders ──
model = pickle.load(open("risk_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ── Location boundaries ──
MIN_LAT = 10.480
MAX_LAT = 10.570
MIN_LON = 76.180
MAX_LON = 76.260


# ── Load, encode and predict ──
def get_risk_df():
    df = pd.read_csv("data.csv")

    df = df[
        (df["Latitude"].between(MIN_LAT, MAX_LAT)) &
        (df["Longitude"].between(MIN_LON, MAX_LON))
    ]

    encoded_df = df.copy()
    for col, le in encoders.items():
        encoded_df[col] = le.transform(encoded_df[col])

    X = encoded_df[
        ["Crime_Count", "Street_Light", "CCTV",
         "Police_Patrol", "Isolation_Level", "Time_Period"]
    ]

    df["Risk_Score"] = model.predict(X)

    return df


# ── Routes ──
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    selected_time = request.args.get("time", "All")

    df = get_risk_df()

    # Trend analysis (before filtering)
    trend_summary = df.groupby("Time_Period")["Risk_Score"].mean().reset_index()
    most_risky_period = trend_summary.loc[
        trend_summary["Risk_Score"].idxmax()
    ]["Time_Period"]

    highest_avg_risk = round(trend_summary["Risk_Score"].max(), 2)

    # Apply filter
    if selected_time != "All":
        df = df[df["Time_Period"] == selected_time]

    # Risk levels
    df["Risk_Level"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    # Counts
    low_count = (df["Risk_Level"] == "Low").sum()
    medium_count = (df["Risk_Level"] == "Medium").sum()
    high_count = (df["Risk_Level"] == "High").sum()

    # Top risky zones
    top_zones = df.sort_values("Risk_Score", ascending=False).head(3)
    top_zones_list = top_zones[
        ["Area_Name", "Risk_Score", "Time_Period"]
    ].to_dict(orient="records")

    # ── Heatmap ──
    m = folium.Map(location=[10.5276, 76.2144], zoom_start=14)

    heat_data = [
        [row["Latitude"], row["Longitude"], row["Risk_Score"]]
        for _, row in df.iterrows()
    ]

    HeatMap(heat_data, radius=30).add_to(m)

    os.makedirs("static", exist_ok=True)
    m.save("static/heatmap.html")

    return render_template(
        "dashboard.html",
        selected_time=selected_time,
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        most_risky_period=most_risky_period,
        highest_avg_risk=highest_avg_risk,
        top_zones=top_zones_list
    )


if __name__ == "__main__":
    app.run(debug=True)

    