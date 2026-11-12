"""
Project 10: Bayesian Probability for Defect Prediction
Manufacturing Defect Prediction System using a Naive Bayes classifier.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Bayesian Defect Predictor",
    page_icon="🏭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 1. Manufacturing training dataset (24 past inspections)
# ---------------------------------------------------------------------------
data = {
    "Temperature": [
        "High", "High", "High", "Normal",
        "High", "High", "Normal", "High",
        "Normal", "High", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "High", "Normal", "Normal", "High",
        "Normal", "Normal", "High", "Normal",
    ],
    "Pressure": [
        "Low", "Low", "Normal", "Low",
        "Low", "Normal", "Normal", "Low",
        "Normal", "Low", "Low", "Normal",
        "Normal", "Normal", "Low", "Normal",
        "Low", "Normal", "Normal", "Normal",
        "Low", "Normal", "Low", "Normal",
    ],
    "Vibration": [
        "High", "High", "High", "High",
        "Normal", "High", "Normal", "High",
        "Normal", "High", "High", "Normal",
        "Normal", "Normal", "High", "Normal",
        "High", "Normal", "Normal", "High",
        "Normal", "Normal", "High", "Normal",
    ],
    "Defect": [
        "Yes", "Yes", "Yes", "No",
        "Yes", "Yes", "No", "Yes",
        "No", "Yes", "No", "No",
        "No", "No", "Yes", "No",
        "Yes", "No", "No", "No",
        "No", "No", "Yes", "No",
    ],
}
df = pd.DataFrame(data)

FEATURES = ["Temperature", "Pressure", "Vibration"]
CLASSES = ["Yes", "No"]  # Yes = Defective, No = Non-defective
LABEL = {"Yes": "Defective", "No": "Non-Defective"}

# Sensor thresholds used to turn raw readings into categories
TEMP_LIMIT = 80.0       # °C   -> above this is "High"
PRESSURE_LIMIT = 4.0    # bar  -> below this is "Low"
VIBRATION_LIMIT = 7.0   # mm/s -> above this is "High"


# ---------------------------------------------------------------------------
# 2. Bayesian model (Naive Bayes with Laplace smoothing)
# ---------------------------------------------------------------------------
def train_model(train_df, alpha=1):
    """Learn prior P(class) and likelihoods P(feature=value | class)."""
    n = len(train_df)
    priors = {c: (train_df["Defect"] == c).sum() / n for c in CLASSES}

    likelihoods = {}
    for f in FEATURES:
        values = sorted(df[f].unique())
        k = len(values)
        for c in CLASSES:
            subset = train_df[train_df["Defect"] == c]
            for v in values:
                count = (subset[f] == v).sum()
                # Laplace smoothing avoids zero probabilities
                likelihoods[(f, v, c)] = (count + alpha) / (len(subset) + alpha * k)
    return priors, likelihoods


def predict(priors, likelihoods, evidence):
    """Return unnormalised scores and posterior probabilities for each class."""
    scores = {}
    for c in CLASSES:
        score = priors[c]
        for f, v in evidence.items():
            score *= likelihoods[(f, v, c)]
        scores[c] = score
    total = sum(scores.values())
    posteriors = {c: scores[c] / total for c in CLASSES}
    return scores, posteriors


def leave_one_out_accuracy(threshold):
    """Test the model on each row after training on the other 23 rows."""
    results = []
    for i in range(len(df)):
        train_df = df.drop(index=i)
        p, l = train_model(train_df)
        evidence = {f: df.loc[i, f] for f in FEATURES}
        _, post = predict(p, l, evidence)
        predicted = "Yes" if post["Yes"] >= threshold else "No"
        results.append((df.loc[i, "Defect"], predicted))
    res = pd.DataFrame(results, columns=["Actual", "Predicted"])
    accuracy = (res["Actual"] == res["Predicted"]).mean()
    confusion = pd.crosstab(
        res["Actual"].map(LABEL), res["Predicted"].map(LABEL),
        rownames=["Actual"], colnames=["Predicted"],
    )
    return accuracy, confusion


priors, likelihoods = train_model(df)

if "log" not in st.session_state:
    st.session_state.log = []

# ---------------------------------------------------------------------------
# 3. Sidebar - input parameters
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Machine Readings")
mode = st.sidebar.radio(
    "Input mode",
    ["Sensor values", "Categories"],
    help="Sensor values are converted to High/Normal/Low using the thresholds shown.",
)

if mode == "Sensor values":
    temp_val = st.sidebar.slider("🌡️ Temperature (°C)", 20.0, 150.0, 85.0, 0.5)
    pres_val = st.sidebar.slider("🔧 Pressure (bar)", 0.0, 10.0, 3.5, 0.1)
    vib_val = st.sidebar.slider("📳 Vibration (mm/s)", 0.0, 20.0, 8.0, 0.1)

    temperature = "High" if temp_val > TEMP_LIMIT else "Normal"
    pressure = "Low" if pres_val < PRESSURE_LIMIT else "Normal"
    vibration = "High" if vib_val > VIBRATION_LIMIT else "Normal"

    st.sidebar.caption(
        f"Thresholds: Temperature > {TEMP_LIMIT:g} °C = High · "
        f"Pressure < {PRESSURE_LIMIT:g} bar = Low · "
        f"Vibration > {VIBRATION_LIMIT:g} mm/s = High"
    )
else:
    temp_val = pres_val = vib_val = None
    temperature = st.sidebar.selectbox("🌡️ Temperature", ["High", "Normal"])
    pressure = st.sidebar.selectbox("🔧 Pressure", ["Low", "Normal"])
    vibration = st.sidebar.selectbox("📳 Vibration", ["High", "Normal"])

threshold = st.sidebar.slider(
    "Decision threshold",
    0.10, 0.90, 0.50, 0.05,
    help="A product is flagged Defective when P(Defective | readings) ≥ this value.",
)

evidence = {"Temperature": temperature, "Pressure": pressure, "Vibration": vibration}
scores, posteriors = predict(priors, likelihoods, evidence)
p_defect = posteriors["Yes"]
verdict = "Defective" if p_defect >= threshold else "Non-Defective"

# ---------------------------------------------------------------------------
# 4. Main page
# ---------------------------------------------------------------------------
st.title("🏭 Manufacturing Defect Prediction System")
st.caption("Bayesian probability model for industrial quality inspection")

tab_predict, tab_bayes, tab_data, tab_log = st.tabs(
    ["🔍 Prediction", "🧮 Bayes Theorem", "📊 Training Data & Model", "📝 Inspection Log"]
)

# ---- Tab 1: Prediction ----------------------------------------------------
with tab_predict:
    c1, c2, c3 = st.columns(3)
    c1.metric("Temperature", temperature, f"{temp_val:g} °C" if temp_val is not None else None,
              delta_color="off")
    c2.metric("Pressure", pressure, f"{pres_val:g} bar" if pres_val is not None else None,
              delta_color="off")
    c3.metric("Vibration", vibration, f"{vib_val:g} mm/s" if vib_val is not None else None,
              delta_color="off")

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Prior P(Defective)", f"{priors['Yes']:.1%}",
              help="Defect rate in past inspections, before looking at the readings.")
    m2.metric("Posterior P(Defective | readings)", f"{p_defect:.1%}",
              f"{(p_defect - priors['Yes']) * 100:+.1f} pts vs prior")
    m3.metric("Posterior P(Non-Defective | readings)", f"{posteriors['No']:.1%}")

    st.progress(p_defect, text=f"Defect probability: {p_defect:.1%}")

    if verdict == "Defective":
        st.error(f"### ❌ Prediction: DEFECTIVE PRODUCT  \n"
                 f"Defect probability {p_defect:.1%} is at or above the {threshold:.0%} threshold. "
                 f"Send this product for re-inspection.")
    else:
        st.success(f"### ✅ Prediction: NON-DEFECTIVE PRODUCT  \n"
                   f"Defect probability {p_defect:.1%} is below the {threshold:.0%} threshold. "
                   f"Product passes quality inspection.")

    if st.button("➕ Add this inspection to the log"):
        st.session_state.log.append({
            "Inspection #": len(st.session_state.log) + 1,
            "Temperature": temperature,
            "Pressure": pressure,
            "Vibration": vibration,
            "P(Defective)": round(p_defect, 4),
            "Prediction": verdict,
        })
        st.toast("Inspection added to log")

    left, right = st.columns(2)
    with left:
        st.subheader("Prior vs Posterior")
        chart_df = pd.DataFrame(
            {"Defective": [priors["Yes"], posteriors["Yes"]],
             "Non-Defective": [priors["No"], posteriors["No"]]},
            index=["1. Prior", "2. Posterior"],
        )
        st.bar_chart(chart_df, stack=False)

    with right:
        st.subheader("Step-by-step calculation")
        rows = []
        for c in CLASSES:
            row = {"Class": LABEL[c], "Prior": priors[c]}
            for f, v in evidence.items():
                row[f"P({f}={v} | class)"] = likelihoods[(f, v, c)]
            row["Prior × Likelihoods"] = scores[c]
            row["Posterior"] = posteriors[c]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Class").style.format("{:.4f}"))

        st.latex(
            r"P(\text{Defective}\mid E)=\frac{%.4f}{%.4f+%.4f}=%.4f"
            % (scores["Yes"], scores["Yes"], scores["No"], p_defect)
        )
        st.caption("E = the observed readings. Each score = prior × product of the "
                   "likelihoods (Naive Bayes assumes the readings are independent given the class).")

# ---- Tab 2: Bayes theorem calculator --------------------------------------
with tab_bayes:
    st.subheader("Bayes' Theorem: updating probability with evidence")
    st.markdown(
        "- **A** = product is defective\n"
        "- **B** = machine shows abnormal readings\n"
        "- **Prior** P(A) = probability of a defect before inspection\n"
        "- **Posterior** P(A|B) = probability of a defect after seeing the readings"
    )
    st.latex(r"P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B\mid A)\,P(A)+P(B\mid \neg A)\,P(\neg A)}")

    b1, b2, b3 = st.columns(3)
    pA = b1.number_input("P(A): prior defect rate", 0.0, 1.0, 0.2, 0.01)
    pBGivenA = b2.number_input("P(B|A): abnormal reading if defective", 0.0, 1.0, 0.8, 0.01)
    pBGivenNotA = b3.number_input("P(B|¬A): abnormal reading if good", 0.0, 1.0, 0.2, 0.01)

    numerator = pBGivenA * pA
    denominator = numerator + pBGivenNotA * (1 - pA)
    if denominator == 0:
        st.warning("P(B) is 0, so the posterior is undefined. Change the inputs.")
    else:
        pAGivenB = numerator / denominator
        st.latex(
            r"P(A\mid B)=\frac{%.2f\times%.2f}{(%.2f\times%.2f)+(%.2f\times%.2f)}=%.4f"
            % (pBGivenA, pA, pBGivenA, pA, pBGivenNotA, 1 - pA, pAGivenB)
        )
        r1, r2 = st.columns(2)
        r1.metric("Prior P(A)", f"{pA:.1%}")
        r2.metric("Posterior P(A|B)", f"{pAGivenB:.1%}", f"{(pAGivenB - pA) * 100:+.1f} pts")
        st.info(f"After seeing abnormal readings, the probability that the product is "
                f"defective changes from **{pA:.0%}** to **{pAGivenB:.1%}**.")

# ---- Tab 3: Training data and model ---------------------------------------
with tab_data:
    d1, d2 = st.columns([3, 2])
    with d1:
        st.subheader(f"Training dataset ({len(df)} past inspections)")
        st.dataframe(df, height=400)
    with d2:
        st.subheader("Class counts")
        counts = df["Defect"].map(LABEL).value_counts()
        st.bar_chart(counts)
        st.write(f"Prior P(Defective) = {counts['Defective']}/{len(df)} = **{priors['Yes']:.3f}**")
        st.write(f"Prior P(Non-Defective) = {counts['Non-Defective']}/{len(df)} = **{priors['No']:.3f}**")

    st.subheader("Learned likelihoods P(reading | class)")
    lk_rows = []
    for f in FEATURES:
        for v in sorted(df[f].unique()):
            lk_rows.append({
                "Parameter": f,
                "Value": v,
                "P(value | Defective)": likelihoods[(f, v, "Yes")],
                "P(value | Non-Defective)": likelihoods[(f, v, "No")],
            })
    st.dataframe(
        pd.DataFrame(lk_rows).style.format(
            {"P(value | Defective)": "{:.3f}", "P(value | Non-Defective)": "{:.3f}"}
        ),
        hide_index=True,
    )
    st.caption("Laplace smoothing: (count + 1) / (class total + number of values).")

    st.subheader("Model evaluation (leave-one-out)")
    acc, conf = leave_one_out_accuracy(threshold)
    e1, e2 = st.columns([1, 2])
    e1.metric("Accuracy", f"{acc:.1%}")
    e2.dataframe(conf)
    st.caption("Each inspection is predicted by a model trained on the other 23, "
               "so the model never sees the row it is tested on.")

# ---- Tab 4: Inspection log -------------------------------------------------
with tab_log:
    st.subheader("Inspection log")
    if st.session_state.log:
        log_df = pd.DataFrame(st.session_state.log)
        st.dataframe(log_df, hide_index=True)
        n_def = (log_df["Prediction"] == "Defective").sum()
        st.write(f"**{n_def}** of **{len(log_df)}** inspected products flagged as defective.")
        st.download_button(
            "⬇️ Download log as CSV",
            log_df.to_csv(index=False).encode("utf-8"),
            file_name="inspection_log.csv",
            mime="text/csv",
        )
        if st.button("🗑️ Clear log"):
            st.session_state.log = []
            st.rerun()
    else:
        st.info("No inspections logged yet. Go to the Prediction tab and click "
                "**Add this inspection to the log**.")
