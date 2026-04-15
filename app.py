from flask import Flask, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# ── Load ML model and encoders ──
model = pickle.load(open("risk_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# ── Location boundaries (Thrissur area) ──
MIN_LAT = 10.480
MAX_LAT = 10.570
MIN_LON = 76.180
MAX_LON = 76.260


# ── Load, encode and predict risk ──
def get_risk_df():
    df = pd.read_csv("data.csv")

    # Filter valid area
    df = df[
        (df["Latitude"].between(MIN_LAT, MAX_LAT)) &
        (df["Longitude"].between(MIN_LON, MAX_LON))
    ]

    # Encode categorical columns
    encoded_df = df.copy()
    for col, le in encoders.items():
        encoded_df[col] = le.transform(encoded_df[col])

    # Features for prediction
    X = encoded_df[
        ["Crime_Count", "Street_Light", "CCTV",
         "Police_Patrol", "Isolation_Level", "Time_Period"]
    ]

    # Predict risk score
    df["Risk_Score"] = model.predict(X)

    return df


# ── Routes ──
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)