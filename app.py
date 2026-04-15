from flask import Flask, render_template, request
import pandas as pd
import pickle

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

    # Optional filter
    if selected_time != "All":
        df = df[df["Time_Period"] == selected_time]

    # Convert to simple list (for future use)
    data = df[["Area_Name", "Risk_Score", "Time_Period"]].to_dict(orient="records")

    return render_template("dashboard.html",
                           data=data,
                           selected_time=selected_time)


if __name__ == "__main__":
    app.run(debug=True)