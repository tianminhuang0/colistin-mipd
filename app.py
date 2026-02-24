import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

# ==========================================
# 0. 页面全局配置与免滚动 CSS 优化
# ==========================================
st.set_page_config(page_title="Colistin MIPD", layout="wide", page_icon="💊")

# 强制压缩页面上下留白，缩小指标字体，确保一屏展示
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    h1 {
        font-size: 1.8rem !important;
        padding-bottom: 0.5rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# 严格遵循学位论文格式规范
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
# 1. 核心模型参数 (靶值 40.53)
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

def simulate_pk_curve(loading_万IU, maint_万IU, interval_h, CL, days=3):
    """二室模型精确数值积分，支持设定首剂加倍"""
    k10 = CL / V1
    k12 = Q / V1
    k21 = Q / V2
    t_inf = 1.0 
    n_doses = int(days * 24 / interval_h)
    
    t_list, A1_list = [], []
    A1, A2, t_offset = 0.0, 0.0, 0.0
    
    for i in range(n_doses):
        dose = loading_万IU if i == 0 else maint_万IU
        rate = dose / t_inf
        
        def ode_on(t, y): return [rate - (k10 + k12)*y[0] + k21*y[1], k12*y[0] - k21*y[1]]
        sol1 = solve_ivp(ode_on, (0, t_inf), [A1, A2], t_eval=np.linspace(0, t_inf, 30))
        A1, A2 = sol1.y[0, -1], sol1.y[1, -1]
        
        t_off = interval_h - t_inf
        def ode_off(t, y): return [-(k10 + k12)*y[0] + k21*y[1], k12*y[0] - k21*y[1]]
        sol2 = solve_ivp(ode_off, (0, t_off), [A1, A2], t_eval=np.linspace(0, t_off, 60))
        A1, A2 = sol2.y[0, -1], sol2.y[1, -1]
        
        t_list.extend([sol1.t + t_offset, sol2.t + t_offset + t_inf])
        A1_list.extend([sol1.y[0], sol2.y[0]])
        t_offset += interval_h
        
    t_arr = np.concatenate(t_list)
    conc = np.concatenate(A1_list) * 1000 / V1 # ng/mL
    return t_arr, conc

# ==========================================
# 3. Web 交互界面 (三列布局设计)
# ==========================================
st.title("💊 硫酸黏菌素重症个体化精准给药系统 (MIPD)")
st.markdown(f"**驱动引擎:** PopPK (二室) + 动态 PK/PD 靶值 (AUC/MIC $\ge$ {TARGET_AUC_MIC}) + Logistic 疗效预测")
st.divider()

# ⭐ 核心改动：采用比例为 1.2 : 1.3 : 2.5 的三列布局
col1, col2, col3 = st.columns([1.2, 1.3, 2.5], gap="large")

with col1:
    st.subheader("📋 患者基线数据录入")
    crcl = st.number_input("肌酐清除率 CrCL (mL/min)", min_value=5.0, max_value=200.0, value=50.0, step=5.0)
    mic = st.selectbox("致病菌 MIC (μg/mL)", options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], index=2)
    apache = st.slider("APACHE II 评分", min_value=0, max_value=50, value=15, step=1)
    
    st.caption("免责声明：本辅助决策系统基于群体药代动力学及真实世界多因素回归模型开发，仅供医学科研展示，不作为法定处方依据。")

# 主流程计算
CL_ind = calc_clearance(crcl)
target_daily_dose_MU = (TARGET_AUC_MIC * CL_ind * mic) / 100

with col2:
    st.subheader("💡 药代参数与推荐方案")
    
    # 将指标横向并排以节省垂直空间
    m1, m2 = st.columns(2)
    m1.metric(label="预测清除率(CL)", value=f"{CL_ind:.2f} L/h")
    m2.metric(label="达标所需日剂量", value=f"{target_daily_dose_MU:.2f} MU")
    
    alert_msg = ""
    # 方案匹配逻辑 
    if target_daily_dose_MU <= 1.25:
        rec_maint, rec_tau, rec_label = 50, 12, "0.5 MU q12h"
        if target_daily_dose_MU < 1.0:
            alert_msg = f"⚠️ **警示：** 理论日剂量（{target_daily_dose_MU:.2f} MU）属于非常规临床低剂量。本预测模型在极端低清除率下的预测能力可能受限，且极低剂量易导致组织浓度不足，请结合临床实际慎重参考。"
            
    elif target_daily_dose_MU <= 1.75:
        rec_maint, rec_tau, rec_label = 75, 12, "0.75 MU q12h"
        
    elif target_daily_dose_MU <= 2.25:
        rec_maint, rec_tau, rec_label = 100, 12, "1.0 MU q12h"
        
    elif target_daily_dose_MU <= 2.75:
        rec_maint, rec_tau, rec_label = 75, 8, "0.75 MU q8h"
        
    else:
        rec_maint, rec_tau, rec_label = 100, 8, "1.0 MU q8h"
        if target_daily_dose_MU > 3.0:
            alert_msg = f"⚠️ **警示：** 理论日剂量（{target_daily_dose_MU:.2f} MU）属于非常规临床极高剂量。本预测模型在极端高清除率下的预测能力可能受限，且超高剂量面临严峻的肾毒性风险，需慎重参考或积极考虑联合靶向用药。"

    actual_daily_dose_万IU = rec_maint * (24 / rec_tau)
    achieved_auc_mic = actual_daily_dose_万IU / (CL_ind * mic)
    rec_prob = predict_efficacy(apache, achieved_auc_mic)
    
    st.success(f"**推荐维持方案：{rec_label}** \n*(建议首剂加倍至 {rec_maint*2/100:.2f} MU)*")
    if alert_msg:
        st.warning(alert_msg)

    st.write(f"**模型预测临床有效率 (基于 APACHE II):** {rec_prob*100:.1f}%")
    st.progress(float(rec_prob))
    if rec_prob < 0.6:
        st.error("🚨 预警：受 APACHE II 危重评分拖累或剂量限制，当前方案预测临床治愈率较低，建议密切监测器官功能。")

with col3:
    st.subheader("📊 首剂加倍方案的 AUC$_{24h}$ 动态达标模拟")

    # 模拟 72 小时的浓度曲线
    t_no, conc_no = simulate_pk_curve(rec_maint, rec_maint, rec_tau, CL_ind, days=3)
    t_ld, conc_ld = simulate_pk_curve(rec_maint*2, rec_maint, rec_tau, CL_ind, days=3)

    # 精确积分计算每一天的 AUC24h
    auc24_no = []
    auc24_ld = []
    for d in range(3):
        m_no = (t_no >= d*24) & (t_no <= (d+1)*24)
        auc24_no.append(np.trapz(conc_no[m_no], t_no[m_no]))
        
        m_ld = (t_ld >= d*24) & (t_ld <= (d+1)*24)
        auc24_ld.append(np.trapz(conc_ld[m_ld], t_ld[m_ld]))

    # 绘制柱状图 (调整 figsize 高度以适配屏幕)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(3)
    width = 0.35

    rects1 = ax.bar(x - width/2, auc24_no, width, label='No Loading Dose', color='#90CAF9')
    rects2 = ax.bar(x + width/2, auc24_ld, width, label='With Loading Dose', color='#1565C0')

    # 靶值水平线
    target_auc24 = TARGET_AUC_MIC * mic
    ax.axhline(target_auc24, color='#C62828', linestyle='--', lw=1.5, label=f'Target AUC$_{{24h}}$ ({target_auc24:.1f} mg·h/L)')

    # 柱子上标数字
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

    # st.pyplot 渲染图像
    st.pyplot(fig)