import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

# ==========================================
# 0. Global page configuration and UI styling
# ==========================================
st.set_page_config(page_title="Colistin Sulfate MIPD", layout="wide", page_icon="\U0001F48A")

# Compress vertical padding so the whole app fits on one screen
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    h1 { font-size: 1.8rem !important; padding-bottom: 0.5rem !important; }
    h3 { font-size: 1.2rem !important; }
    </style>
""", unsafe_allow_html=True)

# Figure formatting (10.5 pt, Times New Roman, non-bold) per thesis style guide
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 10.5
matplotlib.rcParams['axes.titlesize'] = 10.5
matplotlib.rcParams['axes.labelsize'] = 10.5
matplotlib.rcParams['xtick.labelsize'] = 10.5
matplotlib.rcParams['ytick.labelsize'] = 10.5
matplotlib.rcParams['legend.fontsize'] = 10.5
matplotlib.rcParams['axes.titleweight'] = 'normal'
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Core model parameters
# ==========================================
THETA_CL = 6.67
THETA_CRCL = 0.648
CRCL_REF = 76.9
V1 = 0.57
V2 = 58.7
Q = 47.1
TARGET_AUC_MIC = 40.53

# ==========================================
# 2. Core algorithm functions
# ==========================================
def calc_clearance(crcl):
    return THETA_CL * ((crcl / CRCL_REF) ** THETA_CRCL)

def predict_efficacy(apache_ii, auc_mic):
    logit = 3.973 - 0.232 * apache_ii + 0.031 * auc_mic
    prob = 1 / (1 + np.exp(-logit))
    return prob

# Analytical two-compartment solution for precise daily AUC
# (avoids slicing error from numerical integration)
def calculate_analytical_auc_daily(loading, maint, tau, day, CL):
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    S_macro = k10 + k12 + k21
    D_macro = np.sqrt(S_macro**2 - 4 * k10 * k21)
    alpha = (S_macro + D_macro) / 2
    beta = (S_macro - D_macro) / 2

    def single_dose_auc(dose, t_start, t_end, t_inf=1.0):
        if t_end <= 0 or t_start >= t_end:
            return 0.0
        R = dose / t_inf
        Aa = (alpha - k21) / (alpha - beta) / V1 * 1000
        Ab = (k21 - beta) / (alpha - beta) / V1 * 1000

        def auc_0_t(t):
            if t <= 0:
                return 0.0
            if t <= t_inf:
                return R * (Aa/alpha * (t + np.exp(-alpha*t)/alpha - 1/alpha) +
                            Ab/beta  * (t + np.exp(-beta*t)/beta  - 1/beta))
            auc_inf = R * (Aa/alpha * (t_inf + np.exp(-alpha*t_inf)/alpha - 1/alpha) +
                           Ab/beta  * (t_inf + np.exp(-beta*t_inf)/beta  - 1/beta))
            Ca = R * Aa / alpha * (1 - np.exp(-alpha*t_inf))
            Cb = R * Ab / beta  * (1 - np.exp(-beta*t_inf))
            ta = t - t_inf
            auc_post = - Ca/alpha * (np.exp(-alpha*ta) - 1) - Cb/beta * (np.exp(-beta*ta) - 1)
            return auc_inf + auc_post

        return auc_0_t(t_end) - auc_0_t(max(0, t_start))

    t_day_start = (day - 1) * 24.0
    t_day_end = day * 24.0
    n_doses_total = int(t_day_end / tau) + 1

    total_auc = 0.0
    for i in range(n_doses_total):
        dose_time = i * tau
        if dose_time >= t_day_end:
            break
        dose_amt = loading if i == 0 else maint
        rel_t_start = t_day_start - dose_time
        rel_t_end = t_day_end - dose_time
        total_auc += single_dose_auc(dose_amt, rel_t_start, rel_t_end)

    return total_auc

# ==========================================
# 3. Web interface (three-column layout)
# ==========================================
st.title("\U0001F48A Colistin Sulfate Model-Informed Precision Dosing (MIPD) for Critically Ill Adults")
st.markdown(f"**Engine:** Two-compartment PopPK + dynamic PK/PD target (AUC/MIC $\\ge$ {TARGET_AUC_MIC}) + logistic-regression efficacy prediction")
st.divider()

col1, col2, col3 = st.columns([1.2, 1.3, 2.5], gap="large")

with col1:
    st.subheader("\U0001F4CB Patient baseline data")
    # Inputs strictly limited to the model's applicable range
    crcl = st.number_input(
        "Creatinine clearance, CrCL (mL/min)",
        min_value=10.0, max_value=150.0, value=50.0, step=5.0,
        help="Cockcroft\u2013Gault: CrCL = [(140 \u2212 age) \u00d7 weight(kg) \u00d7 (0.85 if female)] / "
             "(72 \u00d7 Scr in mg/dL). Convert Scr from \u03bcmol/L to mg/dL by dividing by 88.4."
    )
    st.caption("\u26A0\uFE0F Enter CrCL from the **Cockcroft\u2013Gault** equation \u2014 the covariate used "
               "in model development. eGFR from CKD-EPI/MDRD is not interchangeable.")
    mic = st.selectbox("Pathogen MIC (\u03bcg/mL)", options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], index=2)
    apache = st.slider("APACHE II score", min_value=5, max_value=40, value=15, step=1)

    st.caption("Disclaimer: This decision-support tool was developed from a real-world multivariable regression model. Inputs outside the predefined range may increase extrapolation error. Results are for research demonstration only and are not a substitute for clinical prescribing judgement.")

# Main calculation
CL_ind = calc_clearance(crcl)
target_daily_dose_MU = (TARGET_AUC_MIC * CL_ind * mic) / 100

with col2:
    st.subheader("\U0001F4A1 Pharmacokinetics & recommended regimen")

    m1, m2 = st.columns(2)
    m1.metric(label="Predicted clearance (CL)", value=f"{CL_ind:.2f} L/h")
    m2.metric(label="Daily dose for target", value=f"{target_daily_dose_MU:.2f} MU")

    alert_msg = ""
    # Regimen-matching logic
    if target_daily_dose_MU <= 1.25:
        rec_maint, rec_tau, rec_label = 50, 12, "0.5 MU q12h"
        if target_daily_dose_MU < 1.0:
            alert_msg = f"\u26A0\uFE0F **Limitation note:** The theoretically required daily dose ({target_daily_dose_MU:.2f} MU) is an unusually low dose. The model's predictive ability under very low clearance is limited; please interpret with clinical caution."

    elif target_daily_dose_MU <= 1.75:
        rec_maint, rec_tau, rec_label = 75, 12, "0.75 MU q12h"

    elif target_daily_dose_MU <= 2.25:
        rec_maint, rec_tau, rec_label = 100, 12, "1.0 MU q12h"

    elif target_daily_dose_MU <= 2.75:
        rec_maint, rec_tau, rec_label = 75, 8, "0.75 MU q8h"

    else:
        rec_maint, rec_tau, rec_label = 100, 8, "1.0 MU q8h"
        if target_daily_dose_MU > 3.0:
            alert_msg = f"\u26A0\uFE0F **Limitation note:** The theoretically required daily dose ({target_daily_dose_MU:.2f} MU) exceeds the conventional dosing range. Very high doses carry a substantial risk of acute kidney injury, and extrapolation error increases at extreme values; please interpret with clinical caution."

    actual_daily_dose_unit = rec_maint * (24 / rec_tau)
    achieved_auc_mic = actual_daily_dose_unit / (CL_ind * mic)
    rec_prob = predict_efficacy(apache, achieved_auc_mic)

    st.success(f"**Recommended regimen: {rec_label}** \n*(Double the first dose to {rec_maint*2/100:.2f} MU as a loading dose)*")
    if alert_msg:
        st.warning(alert_msg)

    st.write(f"**Model-predicted clinical efficacy (adjusted for APACHE II):** {rec_prob*100:.1f}%")
    st.progress(float(rec_prob))
    if rec_prob < 0.6:
        st.error("\U0001F6A8 Alert: Owing to baseline severity (APACHE II) or dose-ceiling constraints, the predicted clinical cure rate of this regimen is low.")

with col3:
    st.subheader("\U0001F4CA Simulated AUC$_{24h}$ target attainment with first-dose doubling")

    # Use the analytical solution to compute daily AUC24h precisely
    auc24_no = [calculate_analytical_auc_daily(rec_maint, rec_maint, rec_tau, d, CL_ind) for d in [1, 2, 3]]
    auc24_ld = [calculate_analytical_auc_daily(rec_maint*2, rec_maint, rec_tau, d, CL_ind) for d in [1, 2, 3]]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(3)
    width = 0.35

    rects1 = ax.bar(x - width/2, auc24_no, width, label='No loading dose', color='#90CAF9')
    rects2 = ax.bar(x + width/2, auc24_ld, width, label='With loading dose', color='#1565C0')

    # Target line
    target_auc24 = TARGET_AUC_MIC * mic
    ax.axhline(target_auc24, color='#C62828', linestyle='--', lw=1.5,
               label=f'Target AUC$_{{24h}}$ ({target_auc24:.1f} mg\u00b7h/L)')

    # Annotate values
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5)

    ax.set_ylabel('AUC$_{24h}$ (mg\u00b7h/L)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Day 1\n(0\u201324 h)', 'Day 2\n(24\u201348 h)', 'Day 3\n(48\u201372 h)'])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)
