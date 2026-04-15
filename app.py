from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# ── Location boundaries (Thrissur area) ──
MIN_LAT = 10.480
MAX_LAT = 10.570
MIN_LON = 76.180
MAX_LON = 76.260


# ── Load and filter dataset ──
def get_risk_df():
    df = pd.read_csv("data.csv")

    # Filter only Thrissur area
    df = df[
        (df["Latitude"].between(MIN_LAT, MAX_LAT)) &
        (df["Longitude"].between(MIN_LON, MAX_LON))
    ]

    return df


# ── Routes ──
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)