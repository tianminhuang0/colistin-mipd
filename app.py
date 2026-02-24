import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

# ==========================================
# 0. 页面全局配置与 UI 优化
# ==========================================
st.set_page_config(page_title="Colistin MIPD", layout="wide", page_icon="💊")

# 压缩上下留白，确保一屏展示
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    h1 { font-size: 1.8rem !important; padding-bottom: 0.5rem !important; }
    h3 { font-size: 1.2rem !important; }
    </style>
""", unsafe_allow_html=True)

# 严格遵循学位论文绘图格式规范 (10.5磅, Times New Roman, 不加粗)
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
# 1. 核心模型参数
# ==========================================
THETA_CL = 6.67
THETA_CRCL = 0.648
CRCL_REF = 76.9
V1 = 0.57
V2 = 58.7
Q = 47.1
TARGET_AUC_MIC = 40.53  

# ==========================================
# 2. 核心算法函数
# ==========================================
def calc_clearance(crcl):
    return THETA_CL * ((crcl / CRCL_REF) ** THETA_CRCL)

def predict_efficacy(apache_ii, auc_mic):
    logit = 3.973 - 0.232 * apache_ii + 0.031 * auc_mic
    prob = 1 / (1 + np.exp(-logit))
    return prob

# 使用二室解析解极其精确地计算每日 AUC (解决数值积分切片误差)
def calculate_analytical_auc_daily(loading, maint, tau, day, CL):
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    S_macro = k10 + k12 + k21
    D_macro = np.sqrt(S_macro**2 - 4 * k10 * k21)
    alpha = (S_macro + D_macro) / 2
    beta = (S_macro - D_macro) / 2

    def single_dose_auc(dose, t_start, t_end, t_inf=1.0):
        if t_end <= 0 or t_start >= t_end: return 0.0
        R = dose / t_inf
        Aa = (alpha - k21) / (alpha - beta) / V1 * 1000
        Ab = (k21 - beta) / (alpha - beta) / V1 * 1000
        
        def auc_0_t(t):
            if t <= 0: return 0.0
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
        if dose_time >= t_day_end: break 
        
        dose_amt = loading if i == 0 else maint
        rel_t_start = t_day_start - dose_time
        rel_t_end = t_day_end - dose_time
        total_auc += single_dose_auc(dose_amt, rel_t_start, rel_t_end)
        
    return total_auc

# ==========================================
# 3. Web 交互界面 (三列布局)
# ==========================================
st.title("💊 硫酸黏菌素重症成人个体化精准给药系统 (MIPD)")
st.markdown(f"**驱动引擎:** PopPK (二室) + 动态 PK/PD 靶值 (AUC/MIC $\ge$ {TARGET_AUC_MIC}) + Logistic 疗效预测")
st.divider()

col1, col2, col3 = st.columns([1.2, 1.3, 2.5], gap="large")

with col1:
    st.subheader("📋 患者基线数据录入")
    # ⭐ 严格限定基线特征在模型适用范围内
    crcl = st.number_input("肌酐清除率 CrCL (mL/min)", min_value=10.0, max_value=150.0, value=50.0, step=5.0)
    mic = st.selectbox("致病菌 MIC (μg/mL)", options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], index=2)
    apache = st.slider("APACHE II 评分", min_value=5, max_value=40, value=15, step=1)
    
    st.caption("免责声明：本辅助决策系统基于真实世界多因素回归模型开发，超出预设范围的指标可能增加外推预测误差，本结果仅供医学科研展示，不作为临床最终处方依据。")

# 主流程计算
CL_ind = calc_clearance(crcl)
target_daily_dose_MU = (TARGET_AUC_MIC * CL_ind * mic) / 100

with col2:
    st.subheader("💡 药代参数与推荐方案")
    
    m1, m2 = st.columns(2)
    m1.metric(label="预测清除率(CL)", value=f"{CL_ind:.2f} L/h")
    m2.metric(label="达标所需日剂量", value=f"{target_daily_dose_MU:.2f} MU")
    
    alert_msg = ""
    # 方案匹配逻辑 
    if target_daily_dose_MU <= 1.25:
        rec_maint, rec_tau, rec_label = 50, 12, "0.5 MU q12h"
        if target_daily_dose_MU < 1.0:
            alert_msg = f"⚠️ **局限性警示：** 纯理论推导所需日剂量（{target_daily_dose_MU:.2f} MU）属于非常规低剂量。本模型在极端低清除状态下的预测能力受限，请结合临床慎重参考。"
            
    elif target_daily_dose_MU <= 1.75:
        rec_maint, rec_tau, rec_label = 75, 12, "0.75 MU q12h"
        
    elif target_daily_dose_MU <= 2.25:
        rec_maint, rec_tau, rec_label = 100, 12, "1.0 MU q12h"
        
    elif target_daily_dose_MU <= 2.75:
        rec_maint, rec_tau, rec_label = 75, 8, "0.75 MU q8h"
        
    else:
        rec_maint, rec_tau, rec_label = 100, 8, "1.0 MU q8h"
        if target_daily_dose_MU > 3.0:
            alert_msg = f"⚠️ **局限性警示：** 理论达标所需日剂量（{target_daily_dose_MU:.2f} MU）已超过常规剂量。超大剂量下面临严峻的急性肾损伤风险，且模型向极高极值外推时误差增加，，请结合临床慎重参考。"

    actual_daily_dose_万IU = rec_maint * (24 / rec_tau)
    achieved_auc_mic = actual_daily_dose_万IU / (CL_ind * mic)
    rec_prob = predict_efficacy(apache, achieved_auc_mic)
    
    st.success(f"**推荐方案：{rec_label}** \n*(推荐首剂加倍至 {rec_maint*2/100:.2f} MU)*")
    if alert_msg:
        st.warning(alert_msg)

    st.write(f"**模型预测临床有效率 (基于 APACHE II):** {rec_prob*100:.1f}%")
    st.progress(float(rec_prob))
    if rec_prob < 0.6:
        st.error("🚨 预警：受基线病情 (APACHE II) 拖累或剂量封顶限制，当前方案预期临床治愈率较低。")

with col3:
    st.subheader("📊 首剂加倍方案的 AUC$_{24h}$ 动态达标模拟")

    # ⭐ 调用解析解精确计算每天的 AUC24h
    auc24_no = [calculate_analytical_auc_daily(rec_maint, rec_maint, rec_tau, d, CL_ind) for d in [1, 2, 3]]
    auc24_ld = [calculate_analytical_auc_daily(rec_maint*2, rec_maint, rec_tau, d, CL_ind) for d in [1, 2, 3]]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(3)
    width = 0.35

    rects1 = ax.bar(x - width/2, auc24_no, width, label='No Loading Dose', color='#90CAF9')
    rects2 = ax.bar(x + width/2, auc24_ld, width, label='With Loading Dose', color='#1565C0')

    # 绘制靶值线
    target_auc24 = TARGET_AUC_MIC * mic
    ax.axhline(target_auc24, color='#C62828', linestyle='--', lw=1.5, label=f'Target AUC$_{{24h}}$ ({target_auc24:.1f} mg·h/L)')

    # 标数值
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5)

    ax.set_ylabel('AUC$_{24h}$ (mg·h/L)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Day 1\n(0-24h)', 'Day 2\n(24-48h)', 'Day 3\n(48-72h)'])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    st.pyplot(fig)
