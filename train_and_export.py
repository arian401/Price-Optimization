"""
Train a logistic‑regression price‑sensitivity model using **live data**
pulled from a production (PostgreSQL) database, and export artefacts
for deployment.

Configuration
-------------
Supply DB credentials via environment variables **or** a local `.env` file:

    DB_USER=your_username
    DB_PASS=your_password
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=your_database

Required packages
-----------------
pip install pandas scikit-learn sqlalchemy psycopg2-binary python-dotenv joblib

Run
----
    python train_and_export.py

The script prints validation metrics and writes:

    models/model.pkl
    models/scaler.pkl
    models/numeric_columns.json

"""

import os
import json
import joblib
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# ───────────────────── 1. Load DB credentials ──────────────────────
load_dotenv()  # reads .env if present

DB_USER = os.getenv("DB_USER", "your_username")
DB_PASS = os.getenv("DB_PASS", "your_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "your_database")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ───────────────────── 2. Pull the data ────────────────────────────
QUERY = """
SELECT
    customer_id,
    total_spent,
    avg_order_value,
    avg_purchase_frequency,
    days_since_last_purchase,
    discount_behavior,
    loyalty_program_member,
    days_in_advance,
    flight_type,
    cabin_class,
    will_buy_after_price_increase
FROM customer_data;
"""

print("⏳ Pulling data from database …")
df = pd.read_sql(QUERY, con=engine)
print(f"✅ Pulled {len(df):,} rows")

# ───────────────────── 3. Feature lists ────────────────────────────
numeric_features = [
    "total_spent",
    "avg_order_value",
    "avg_purchase_frequency",
    "days_since_last_purchase",
    "discount_behavior",
    "loyalty_program_member",
    "days_in_advance",
]

categorical_features = ["flight_type", "cabin_class"]

X = df[numeric_features + categorical_features]
y = df["will_buy_after_price_increase"].astype(int)

# ───────────────────── 4. Train / test split ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ───────────────────── 5. Preprocessing + model ────────────────────
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse=False), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        (
            "clf",
            LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="liblinear"
            ),
        ),
    ]
)

model.fit(X_train, y_train)

# ───────────────────── 6. Metrics ──────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Validation report\n", classification_report(y_test, y_pred))
print("ROC‑AUC:", round(roc_auc_score(y_test, y_proba), 3))

# ───────────────────── 7. Save artefacts ───────────────────────────
Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/model.pkl")

# Save the scaler separately (optional, but handy)
scaler = model.named_steps["preprocess"].named_transformers_["num"]
joblib.dump(scaler, "models/scaler.pkl")

with open("models/numeric_columns.json", "w") as f:
    json.dump(numeric_features, f, indent=2)

print("✅ Saved model.pkl, scaler.pkl, numeric_columns.json in /models")
