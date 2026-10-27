import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Startup Success Predictor", page_icon="🚀", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "funding_rounds", "relationships", "age_last_funding_year",
    "milestones", "avg_participants", "funding_total_usd"
]
FEATURE_LABELS = {
    "funding_rounds": "Total Funding Rounds",
    "relationships": "Key Professional Relationships",
    "age_last_funding_year": "Startup Age at Last Funding (Years)",
    "milestones": "Major Milestones Achieved",
    "avg_participants": "Average Investor Participants",
    "funding_total_usd": "Total Funding Raised (USD)",
}
MODEL_FILES = {"SVM Model": "model.pkl", "Best Model": "best_model.pkl"}

@st.cache_data(show_spinner="Loading the startup dataset automatically...")
def load_dataset():
    """Download the same startup_data.csv source referenced in the project notebook."""
    api_url = "https://api.github.com/repos/iansind/startup-success/contents/"
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    csv_files = [item for item in response.json() if item.get("name", "").lower().endswith(".csv")]
    if not csv_files:
        raise RuntimeError("No CSV dataset was found in the project's GitHub repository.")
    selected = next((item for item in csv_files if item["name"] == "startup_data.csv"), csv_files[0])
    csv_response = requests.get(selected["download_url"], timeout=60)
    csv_response.raise_for_status()
    from io import BytesIO
    data = pd.read_csv(BytesIO(csv_response.content))
    data = data.drop_duplicates().reset_index(drop=True)
    return data

@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

@st.cache_resource
def make_scaler(data):
    """Load the original fitted scaler supplied by the user and fit label mapping from dataset."""
    missing = [col for col in FEATURES + ["status"] if col not in data.columns]
    if missing:
        raise ValueError("Dataset is missing expected columns: " + ", ".join(missing))
    scaler_path = BASE_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.pkl is missing. Keep the original fitted scaler beside app.py.")
    scaler = joblib.load(scaler_path)
    if getattr(scaler, "n_features_in_", len(FEATURES)) != len(FEATURES):
        raise ValueError("The supplied scaler does not match the six expected features.")
    encoder = LabelEncoder()
    encoder.fit(data["status"].fillna("Unknown").astype(str))
    return scaler, encoder

st.title("🚀 Startup Success Prediction Dashboard")
st.caption("Enter startup metrics to predict its outcome. The dataset is loaded automatically; you do not need to upload a CSV.")

try:
    df = load_dataset()
    scaler, label_encoder = make_scaler(df)
except Exception as exc:
    st.error("Could not automatically load the training dataset. Check your internet connection and try again.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Startup Metrics Input")
    model_name = st.selectbox("Prediction model", list(MODEL_FILES.keys()))
    funding_rounds = st.number_input("Total Funding Rounds", min_value=0, max_value=100, value=3, step=1)
    relationships = st.number_input("Key Professional Relationships", min_value=0, max_value=10000, value=10, step=1)
    age_last_funding_year = st.number_input("Startup Age at Last Funding (Years)", min_value=-30.0, max_value=100.0, value=5.0, step=0.5)
    milestones = st.number_input("Major Milestones Achieved", min_value=0, max_value=100, value=2, step=1)
    avg_participants = st.number_input("Average Investor Participants", min_value=0.0, max_value=1000.0, value=3.0, step=0.5)
    funding_total_usd = st.number_input("Total Funding Raised (USD)", min_value=0.0, max_value=1_000_000_000_000.0, value=1_000_000.0, step=100_000.0, format="%.2f")
    predict = st.button("🔮 Predict Startup Outcome", type="primary", use_container_width=True)

input_values = {
    "funding_rounds": funding_rounds,
    "relationships": relationships,
    "age_last_funding_year": age_last_funding_year,
    "milestones": milestones,
    "avg_participants": avg_participants,
    "funding_total_usd": funding_total_usd,
}
input_df = pd.DataFrame([input_values], columns=FEATURES)

model_path = BASE_DIR / MODEL_FILES[model_name]
if not model_path.exists():
    st.error(f"Missing {MODEL_FILES[model_name]}. Keep both model files in the same folder as app.py.")
    st.stop()
try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error(f"Could not load the selected model: {exc}")
    st.stop()
if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
    st.error(f"The selected model expects {getattr(model, 'n_features_in_', '?')} features, but this app supplies {len(FEATURES)}.")
    st.stop()

scaled_input = scaler.transform(input_df)
# Predict on first page load as well as after every input change; button remains a visible action.
try:
    pred_encoded = int(model.predict(scaled_input)[0])
    class_values = list(getattr(model, "classes_", label_encoder.transform(label_encoder.classes_)))
    if pred_encoded in range(len(label_encoder.classes_)):
        prediction_label = label_encoder.inverse_transform([pred_encoded])[0]
    else:
        prediction_label = str(pred_encoded)
    probabilities = model.predict_proba(scaled_input)[0] if hasattr(model, "predict_proba") else None
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

# Layout inspired by the supplied screenshot: prominent result and probability chart.
left, right = st.columns([1, 1.12], gap="large")
with left:
    st.subheader("📊 Prediction Result")
    normalized_label = str(prediction_label).strip().lower()
    success_like = normalized_label in {"acquired", "ipo", "operating", "active", "success", "successful"}
    if success_like:
        st.success(f"### {str(prediction_label).title()}")
        st.write("The model predicts this startup outcome based on the metrics entered.")
    else:
        st.error(f"### {str(prediction_label).title()}")
        st.write("The model predicts this startup outcome based on the metrics entered.")
    st.caption("This is a machine-learning prediction, not a guarantee of future performance.")
    st.subheader("Current Input Summary")
    a, b = st.columns(2)
    a.metric("Total Funding", f"${funding_total_usd:,.0f}")
    b.metric("Funding Rounds", f"{funding_rounds:,}")
    c, d = st.columns(2)
    c.metric("Relationships", f"{relationships:,}")
    d.metric("Milestones", f"{milestones:,}")

# Explain likely risk signals using comparisons with the dataset. These are associations, not proven causes.
status_text = df["status"].fillna("Unknown").astype(str).str.strip().str.lower()
closed_mask = status_text.str.contains("closed|close|failed|failure|bankrupt|defunct", regex=True)
nonclosed_mask = ~closed_mask

with left:
    if normalized_label in {"closed", "close", "failed", "failure", "bankrupt", "defunct"} or "closed" in normalized_label or "fail" in normalized_label:
        st.subheader("🔎 Possible Closure Risk Signals")
        st.write("The model predicts a closure-related outcome. The points below show how your inputs compare with the dataset; they are **potential warning signals, not confirmed causes**.")
        risk_features = {
            "funding_rounds": "Funding rounds",
            "relationships": "Professional relationships",
            "milestones": "Milestones achieved",
            "avg_participants": "Average investor participants",
            "funding_total_usd": "Total funding raised",
        }
        signals = []
        if nonclosed_mask.any():
            for feature, label in risk_features.items():
                vals = pd.to_numeric(df.loc[nonclosed_mask, feature], errors="coerce").dropna()
                if not vals.empty:
                    median_value = float(vals.median())
                    entered = float(input_values[feature])
                    if median_value > 0 and entered < median_value * 0.8:
                        if feature == "funding_total_usd":
                            signals.append(f"**Lower total funding:** ${entered:,.0f} entered, compared with a dataset median of ${median_value:,.0f} for non-closed startups.")
                        else:
                            signals.append(f"**Fewer {label.lower()}:** {entered:g} entered, compared with a dataset median of {median_value:g} for non-closed startups.")
        if signals:
            for signal in signals:
                st.markdown(f"- {signal}")
        else:
            st.info("No clear below-median warning signal was identified from these inputs. The model combines all six features, so its prediction may reflect patterns that are not obvious from a simple comparison.")
        st.caption("This dashboard cannot determine the actual reason a specific startup closed. The dataset contains outcome labels, not verified explanations for each closure.")
    else:
        st.subheader("🔎 What supports this prediction?")
        st.write("The model predicts a non-closure outcome. This is a statistical estimate, not a guarantee. The model does not identify a verified cause or guarantee that any startup will succeed.")

with right:
    st.subheader("📈 Outcome Probability")
    if probabilities is not None:
        # Use the estimator's class order and decode it with the dataset label encoder.
        labels = []
        for cls in model.classes_:
            try:
                labels.append(str(label_encoder.inverse_transform([int(cls)])[0]).title())
            except Exception:
                labels.append(str(cls))
        prob_df = pd.DataFrame({"Startup Outcome": labels, "Probability (%)": np.asarray(probabilities) * 100})
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.pie(prob_df["Probability (%)"], labels=prob_df["Startup Outcome"], autopct="%1.1f%%", startangle=90, wedgeprops={"width": 0.42})
        ax.set_title("Estimated probability by outcome")
        st.pyplot(fig, use_container_width=True)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
    else:
        st.info("This model does not support probability estimates.")

st.divider()
st.header("💰 How total funding affects this prediction")
st.caption("This sensitivity chart changes only total funding while keeping your other five inputs fixed. It shows how this trained model's estimated probabilities respond; it does not prove that funding causes an outcome.")

try:
    funding_values = pd.to_numeric(df["funding_total_usd"], errors="coerce").dropna()
    funding_values = funding_values[funding_values >= 0]
    if funding_values.empty:
        funding_grid = np.linspace(0, max(float(funding_total_usd) * 2, 1_000_000), 25)
    else:
        low = float(funding_values.quantile(0.05))
        high = float(funding_values.quantile(0.95))
        if low == high:
            low, high = 0.0, max(float(funding_values.max()), 1.0)
        funding_grid = np.linspace(low, high, 35)

    sensitivity_rows = []
    for amount in funding_grid:
        scenario = input_df.copy()
        scenario.loc[0, "funding_total_usd"] = float(amount)
        scaled_scenario = scaler.transform(scenario)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(scaled_scenario)[0]
            for cls, prob in zip(model.classes_, p):
                try:
                    outcome = str(label_encoder.inverse_transform([int(cls)])[0]).title()
                except Exception:
                    outcome = str(cls)
                sensitivity_rows.append({"Total funding (USD)": amount, "Outcome": outcome, "Estimated probability (%)": float(prob) * 100})
        else:
            predicted = model.predict(scaled_scenario)[0]
            try:
                outcome = str(label_encoder.inverse_transform([int(predicted)])[0]).title()
            except Exception:
                outcome = str(predicted)
            sensitivity_rows.append({"Total funding (USD)": amount, "Outcome": outcome, "Estimated probability (%)": np.nan})

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    if sensitivity_df["Estimated probability (%)"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for outcome, group in sensitivity_df.groupby("Outcome"):
            ax.plot(group["Total funding (USD)"], group["Estimated probability (%)"], label=outcome)
        ax.axvline(float(funding_total_usd), linestyle="--", label="Your entered funding")
        ax.set_xlabel("Total funding (USD)")
        ax.set_ylabel("Estimated probability (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Predicted outcome probability vs. total funding")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("If a line changes but the predicted class stays the same, funding is affecting the estimated probability without crossing the model's decision boundary.")
    else:
        st.info("The selected model does not provide probability estimates for this sensitivity chart.")
except Exception as exc:
    st.warning(f"Could not create the funding sensitivity chart: {exc}")

st.divider()
st.header("📉 Dataset Visualizations")
st.caption("These charts describe the full dataset loaded automatically from the same source used in the notebook; they are not fabricated from the current input.")
chart1, chart2 = st.columns(2)
with chart1:
    st.subheader("Startup Status Distribution")
    status_counts = df["status"].fillna("Unknown").astype(str).value_counts().rename_axis("Status").reset_index(name="Number of Startups")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(status_counts["Status"], status_counts["Number of Startups"])
    ax.set_ylabel("Number of startups")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
with chart2:
    st.subheader("Funding Amount Distribution")
    funds = pd.to_numeric(df["funding_total_usd"], errors="coerce").dropna()
    funds = funds[funds >= 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(funds, bins=30, edgecolor="black")
    ax.set_xlabel("Total funding (USD)")
    ax.set_ylabel("Number of startups")
    ax.set_title("Distribution of total funding")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
chart3, chart4 = st.columns(2)
with chart3:
    st.subheader("Top Startup Categories")
    if "category_code" in df.columns:
        cats = df["category_code"].fillna("Unknown").astype(str).value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(cats.index, cats.values)
        ax.set_xlabel("Number of startups")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("The dataset has no category_code column.")
with chart4:
    st.subheader("Feature Correlation")
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES], rotation=70, ha="right")
    ax.set_yticks(range(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES])
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with st.expander("View input values sent to the model"):
    pretty_input = pd.DataFrame({"Metric": [FEATURE_LABELS[c] for c in FEATURES], "Entered value": [input_values[c] for c in FEATURES]})
    st.dataframe(pretty_input, use_container_width=True, hide_index=True)

st.caption("The dashboard loads your original fitted scaler from scaler.pkl. Predictions depend on the trained model and all six input features; the funding sensitivity chart varies only funding while holding the other inputs fixed.")
