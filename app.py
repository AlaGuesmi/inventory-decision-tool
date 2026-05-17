import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import calendar
from datetime import timedelta
 
# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Favorita Inventory Decision Tool",
    page_icon="🛒",
    layout="wide"
)
 
# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Open Sans', sans-serif;
        background-color: #1b2a4a;
        color: #e8edf2;
    }
 
    .stApp {
        background-color: #1b2a4a;
    }
 
    /* Main title */
    h1 {
        font-family: 'Open Sans', sans-serif;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: 0.5px;
    }
 
    /* Section headers */
    h2 {
        font-family: 'Open Sans', sans-serif;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #7ec8e3 !important;
        border-bottom: 2px solid #2e4a6a;
        padding-bottom: 10px;
        margin-top: 28px !important;
    }
 
    /* Subsection headers */
    h3 {
        font-family: 'Open Sans', sans-serif;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #b0d4e8 !important;
    }
 
    /* Normal text */
    p, div, span {
        font-family: 'Open Sans', sans-serif;
        font-size: 1.05rem !important;
        color: #d0dce8;
        line-height: 1.6;
    }
 
    /* Input labels */
    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #7ec8e3 !important;
    }
 
    /* Input boxes */
    .stNumberInput input {
        background-color: #243555 !important;
        color: #ffffff !important;
        border: 1px solid #3a5a80 !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
        padding: 8px !important;
    }
 
    /* Select box */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #243555 !important;
        border: 1px solid #3a5a80 !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
    }
 
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #243555;
        border: 1px solid #3a5a80;
        border-radius: 12px;
        padding: 18px;
    }
 
    [data-testid="metric-container"] label {
        color: #7ec8e3 !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
 
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
 
    /* Divider */
    hr {
        border-color: #2e4a6a !important;
        margin: 24px 0 !important;
    }
 
    /* Dataframe */
    .dataframe th {
        font-size: 1rem !important;
        font-weight: 700 !important;
        background-color: #1b2a4a !important;
        color: #7ec8e3 !important;
        text-align: center !important;
    }
 
    .dataframe td {
        font-size: 1rem !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        return pickle.load(f)
 
@st.cache_data
def load_data():
    df = pd.read_csv('favorita_personal_care.csv', parse_dates=['date'])
    results = pd.read_csv('results.csv', parse_dates=['date'])
    return df, results
 
xgb_model = load_model()
df, results = load_data()
 
# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
FEATURES = ['onpromotion', 'dcoilwtico', 'transactions', 'is_national_holiday',
            'precipitation_mm', 'is_payday', 'day_of_week', 'month', 'year',
            'lag_1', 'lag_7', 'lag_14']
 
LAST_DATE = df['date'].max()
XGB_RESIDUALS = results['residuals_xgb'].values
 
SCENARIO_PERCENTILES = {
    'Optimistic (90% CSL)': 90,
    'Base Case (95% CSL)': 95,
    'Pessimistic (99% CSL)': 99
}
 
SCENARIO_COLORS = {
    'Optimistic (90% CSL)': '#2ecc71',
    'Base Case (95% CSL)': '#5ba8f5',
    'Pessimistic (99% CSL)': '#e74c3c'
}
 
# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def generate_forecast(df, n_days=7):
    last_sales = list(df['sales'].values)
    last_row = df.iloc[-1]
    forecast_dates = []
    forecasts = []
 
    for i in range(1, n_days + 1):
        next_date = LAST_DATE + timedelta(days=i)
        day_of_week = next_date.weekday()
        month = next_date.month
        year = next_date.year
        last_day = calendar.monthrange(year, month)[1]
        is_payday = 1 if next_date.day in [15, last_day] else 0
 
        row = {
            'onpromotion': last_row['onpromotion'],
            'dcoilwtico': last_row['dcoilwtico'],
            'transactions': last_row['transactions'],
            'is_national_holiday': 0,
            'precipitation_mm': last_row['precipitation_mm'],
            'is_payday': is_payday,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'lag_1': last_sales[-1],
            'lag_7': last_sales[-7] if len(last_sales) >= 7 else last_sales[0],
            'lag_14': last_sales[-14] if len(last_sales) >= 14 else last_sales[0]
        }
 
        X = pd.DataFrame([row])[FEATURES]
        pred = xgb_model.predict(X)[0]
        forecasts.append(max(0, pred))
        last_sales.append(pred)
        forecast_dates.append(next_date)
 
    return forecast_dates, forecasts
 
 
def compute_policy(percentile, L, p, K, h_rate):
    h = h_rate * p
    D_annual = results['actual'].mean() * 365
    SS = np.percentile(XGB_RESIDUALS, percentile) * np.sqrt(L)
    mu_L = results['xgboost'].mean() * L
    ROP = mu_L + SS
    EOQ = np.sqrt((2 * D_annual * K) / h)
    TC = (EOQ / 2) * h + (D_annual / EOQ) * K + SS * h
    return {
        'SS': round(SS, 0),
        'ROP': round(ROP, 0),
        'EOQ': round(EOQ, 0),
        'TC': round(TC, 2),
        'CSL': percentile
    }
 
 
def project_inventory(current_inv, forecasts, ROP, EOQ, L):
    inventory = current_inv
    trajectory = [current_inv]
    order_arriving = None
 
    for i, demand in enumerate(forecasts):
        if order_arriving == i:
            inventory += EOQ
        inventory = max(0, inventory - demand)
        trajectory.append(inventory)
        if inventory <= ROP and order_arriving is None:
            order_arriving = i + 1 + L
 
    return trajectory
 
 
def make_plot(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1b2a4a')
    ax.set_facecolor('#1b2a4a')
    ax.tick_params(colors='#b0d4e8', labelsize=11)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#3a5a80')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#3a5a80')
    return fig, ax
 
 
# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div style='text-align: center; padding: 24px 0 12px 0;'>
    <div style='font-size: 2.8rem; font-weight: 800; color: #ffffff; margin-bottom: 6px; font-family: Open Sans, sans-serif;'>
        🛒 Favorita Inventory Decision Tool
    </div>
    <div style='font-size: 1.2rem; color: #7ec8e3; margin-top: 0; font-family: Open Sans, sans-serif;'>
        Quito &nbsp;|&nbsp; Store 44 &nbsp;|&nbsp; Personal Care Category
    </div>
    <div style='font-size: 0.95rem; color: #7a9bbf; margin-top: 6px; font-family: Open Sans, sans-serif;'>
        Powered by XGBoost Demand Forecasting &nbsp;·&nbsp; Continuous Review (s,Q) Policy &nbsp;·&nbsp; Decisions simulated as of August 15, 2017
    </div>
</div>
""", unsafe_allow_html=True)
 
st.divider()
 
# ─────────────────────────────────────────
# MANAGER INPUT
# ─────────────────────────────────────────
st.markdown("## Manager Input")
st.markdown("<p style='color:#b0d4e8; font-size:1.05rem;'>Enter the current inventory level and adjust cost parameters if needed. All inputs use research-validated defaults.</p>", unsafe_allow_html=True)
 
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    current_inventory = st.number_input(
        "Current Inventory (units)",
        min_value=0, value=5000, step=100,
        help="Current number of units physically on hand"
    )
with col2:
    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=1, max_value=21, value=7,
        help="Days between placing and receiving an order"
    )
with col3:
    unit_value = st.number_input(
        "Unit Value ($)",
        min_value=1, max_value=50, value=4, step=1,
        help="Average unit value of Personal Care products"
    )
with col4:
    K = st.number_input(
        "Ordering Cost K ($)",
        min_value=10, max_value=200, value=50, step=5,
        help="Fixed cost per replenishment order placed"
    )
with col5:
    h_pct = st.number_input(
        "Holding Cost Rate (%)",
        min_value=10, max_value=60, value=25, step=5,
        help="Annual holding cost as % of unit value"
    )
    h_rate = h_pct / 100
 
st.divider()
 
# ─────────────────────────────────────────
# GENERATE FORECAST
# ─────────────────────────────────────────
try:
    forecast_dates, forecasts = generate_forecast(df, n_days=7)
except Exception as e:
    st.error(f"Forecast generation error: {e}")
    st.stop()
 
p5 = np.percentile(XGB_RESIDUALS, 5)
p95 = np.percentile(XGB_RESIDUALS, 95)
upper_bound = [max(0, f + p95) for f in forecasts]
lower_bound = [max(0, f + p5) for f in forecasts]
 
# ─────────────────────────────────────────
# 7-DAY DEMAND FORECAST
# ─────────────────────────────────────────
st.markdown("## 7-Day Demand Forecast")
st.markdown("<p style='color:#b0d4e8; font-size:1.05rem;'>Forward-looking demand forecast generated by the XGBoost model. The shaded band represents the 5th–95th percentile uncertainty range derived from historical forecast residuals.</p>", unsafe_allow_html=True)
 
fig1, ax1 = make_plot(figsize=(12, 4))
ax1.plot(forecast_dates, forecasts, 'o-', color='#5ba8f5',
         linewidth=2.5, markersize=8, label='Point Forecast', zorder=3)
ax1.fill_between(forecast_dates, lower_bound, upper_bound,
                 alpha=0.18, color='#5ba8f5', label='Uncertainty Band (5th–95th pct)')
ax1.set_xlabel('Date', color='#b0d4e8', fontsize=12)
ax1.set_ylabel('Forecasted Units Sold', color='#b0d4e8', fontsize=12)
ax1.set_title('7-Day Demand Forecast — Personal Care, Store 44',
              color='#ffffff', fontsize=14, fontweight='bold', pad=15)
ax1.legend(facecolor='#243555', edgecolor='#3a5a80', labelcolor='#e8edf2', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)
 
forecast_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
    'Point Forecast (units)': [int(round(f, 0)) for f in forecasts],
    'Lower Bound (units)': [int(round(l, 0)) for l in lower_bound],
    'Upper Bound (units)': [int(round(u, 0)) for u in upper_bound]
})
st.dataframe(forecast_df, use_container_width=True, hide_index=True)
st.divider()
 
# ─────────────────────────────────────────
# SCENARIO ANALYSIS
# ─────────────────────────────────────────
st.markdown("## Scenario Analysis")
st.markdown("<p style='color:#b0d4e8; font-size:1.05rem;'>Three inventory policies derived simultaneously under different uncertainty assumptions. Select the scenario that best reflects your current operational context.</p>", unsafe_allow_html=True)
 
col_opt, col_base, col_pes = st.columns(3)
with col_opt:
    st.markdown("""
    <div style='background-color:#1e3d2a; border:2px solid #2ecc71; border-radius:12px; padding:18px;'>
        <p style='color:#2ecc71; font-weight:700; font-size:1.1rem; margin-bottom:6px;'>🟢 Optimistic — 90% CSL</p>
        <p style='color:#b2dfdb; font-size:0.98rem; margin:0; line-height:1.5;'>Stable conditions. No upcoming holidays, promotions, or payday events. Lean safety stock, lower holding cost.</p>
    </div>
    """, unsafe_allow_html=True)
with col_base:
    st.markdown("""
    <div style='background-color:#1e2d4a; border:2px solid #5ba8f5; border-radius:12px; padding:18px;'>
        <p style='color:#5ba8f5; font-weight:700; font-size:1.1rem; margin-bottom:6px;'>🔵 Base Case — 95% CSL</p>
        <p style='color:#b3d4f5; font-size:0.98rem; margin:0; line-height:1.5;'>Standard operating conditions. Reflects the research framework validated baseline inventory policy.</p>
    </div>
    """, unsafe_allow_html=True)
with col_pes:
    st.markdown("""
    <div style='background-color:#3d1e1e; border:2px solid #e74c3c; border-radius:12px; padding:18px;'>
        <p style='color:#e74c3c; font-weight:700; font-size:1.1rem; margin-bottom:6px;'>🔴 Pessimistic — 99% CSL</p>
        <p style='color:#f5b3b3; font-size:0.98rem; margin:0; line-height:1.5;'>Elevated demand expected. Upcoming holiday, payday, or promotion. Higher safety stock, maximum protection.</p>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
policies = {}
for scenario, percentile in SCENARIO_PERCENTILES.items():
    policies[scenario] = compute_policy(percentile, lead_time, unit_value, K, h_rate)
 
# Build scenario table as proper dataframe
scenario_records = []
row_bg = ['#1e3d2a', '#1e2d4a', '#3d1e1e']
for i, (scenario, policy) in enumerate(policies.items()):
    decision = '🔴 Place Order' if current_inventory <= policy['ROP'] else '🟢 No Order'
    scenario_records.append({
        'Scenario': scenario,
        'Target CSL (%)': f"{policy['CSL']}%",
        'Safety Stock (units)': f"{int(policy['SS']):,}",
        'Reorder Point (units)': f"{int(policy['ROP']):,}",
        'Order Quantity (units)': f"{int(policy['EOQ']):,}",
        'Annual Cost ($)': f"${policy['TC']:,.2f}",
        'Decision': decision
    })
 
scenario_df = pd.DataFrame(scenario_records)
 
def color_rows(row):
    idx = scenario_df.index.get_loc(row.name)
    colors = [
        'background-color: #1e3d2a; color: #e8edf2; font-size: 1rem; text-align: center; border: 1px solid #2e4a6a;',
        'background-color: #1e2d4a; color: #e8edf2; font-size: 1rem; text-align: center; border: 1px solid #2e4a6a;',
        'background-color: #3d1e1e; color: #e8edf2; font-size: 1rem; text-align: center; border: 1px solid #2e4a6a;'
    ]
    return [colors[idx]] * len(row)
 
styled_scenario = scenario_df.style.apply(color_rows, axis=1).set_table_styles([
    {'selector': 'th', 'props': [
        ('background-color', '#1b2a4a'),
        ('color', '#7ec8e3'),
        ('font-weight', 'bold'),
        ('font-size', '1rem'),
        ('text-align', 'center'),
        ('border', '1px solid #3a5a80'),
        ('padding', '12px')
    ]},
    {'selector': 'td', 'props': [
        ('padding', '12px 16px'),
        ('border', '1px solid #2e4a6a'),
        ('font-size', '1rem')
    ]}
])
 
st.dataframe(styled_scenario, use_container_width=True, hide_index=True)
 
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Projected Inventory Trajectory — Next 7 Days")
 
fig2, ax2 = make_plot(figsize=(12, 4))
all_dates = [LAST_DATE] + forecast_dates
for scenario, policy in policies.items():
    trajectory = project_inventory(
        current_inventory, forecasts,
        policy['ROP'], policy['EOQ'], lead_time
    )
    ax2.plot(all_dates, trajectory, 'o-', color=SCENARIO_COLORS[scenario],
             linewidth=2.5, markersize=7, label=scenario)
 
ax2.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.5, label='Zero Inventory')
ax2.set_xlabel('Date', color='#b0d4e8', fontsize=12)
ax2.set_ylabel('Inventory Level (units)', color='#b0d4e8', fontsize=12)
ax2.set_title('Projected Inventory Trajectory Under Three Scenarios',
              color='#ffffff', fontsize=14, fontweight='bold', pad=15)
ax2.legend(facecolor='#243555', edgecolor='#3a5a80', labelcolor='#e8edf2', fontsize=11)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
st.divider()
 
# ─────────────────────────────────────────
# REPLENISHMENT DECISION
# ─────────────────────────────────────────
st.markdown("## Replenishment Decision")
 
selected_scenario = st.selectbox(
    "Select your operating scenario:",
    options=list(policies.keys()),
    index=1
)
 
selected_policy = policies[selected_scenario]
ROP = selected_policy['ROP']
EOQ = selected_policy['EOQ']
 
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Inventory", f"{current_inventory:,} units")
with col2:
    st.metric("Reorder Point", f"{int(ROP):,} units")
with col3:
    delta = int(current_inventory - ROP)
    st.metric("Buffer above ROP", f"{delta:,} units", delta=delta, delta_color="normal")
 
st.markdown("<br>", unsafe_allow_html=True)
 
if current_inventory <= ROP:
    st.markdown(f"""
    <div style='background-color:#3d1e1e; border:2px solid #e74c3c; border-radius:14px; padding:24px;'>
        <div style='color:#e74c3c; font-weight:800; font-size:1.4rem; margin-bottom:12px; font-family:Open Sans,sans-serif;'>🔴 PLACE ORDER NOW</div>
        <p style='color:#f5c6c6; font-size:1.05rem; margin:8px 0;'><b>Order Quantity:</b> {int(EOQ):,} units</p>
        <p style='color:#f5c6c6; font-size:1.05rem; margin:8px 0;'><b>Expected Delivery:</b> {(LAST_DATE + timedelta(days=lead_time)).strftime('%Y-%m-%d')} ({lead_time} days)</p>
        <p style='color:#f5c6c6; font-size:1.05rem; margin:8px 0;'><b>Projected inventory at delivery:</b> {max(0, int(current_inventory - sum(forecasts[:lead_time]))):,} units</p>
    </div>
    """, unsafe_allow_html=True)
else:
    days_until_rop = None
    inv = current_inventory
    for i, demand in enumerate(forecasts):
        inv -= demand
        if inv <= ROP:
            days_until_rop = i + 1
            break
 
    if days_until_rop:
        st.markdown(f"""
        <div style='background-color:#3d3000; border:2px solid #f39c12; border-radius:14px; padding:24px;'>
            <div style='color:#f39c12; font-weight:800; font-size:1.4rem; margin-bottom:12px; font-family:Open Sans,sans-serif;'>🟡 NO ORDER — MONITOR CLOSELY</div>
            <p style='color:#fdeaa7; font-size:1.05rem; margin:8px 0;'>Inventory above reorder point but may reach it within <b>{days_until_rop} days</b>.</p>
            <p style='color:#fdeaa7; font-size:1.05rem; margin:8px 0;'>Consider switching to the <b>Pessimistic</b> scenario and placing an order soon.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#1e3d2a; border:2px solid #2ecc71; border-radius:14px; padding:24px;'>
            <div style='color:#2ecc71; font-weight:800; font-size:1.4rem; margin-bottom:12px; font-family:Open Sans,sans-serif;'>🟢 NO ORDER NEEDED</div>
            <p style='color:#b2dfdb; font-size:1.05rem; margin:8px 0;'>Current inventory of <b>{current_inventory:,} units</b> is comfortably above the reorder point of <b>{int(ROP):,} units</b>.</p>
            <p style='color:#b2dfdb; font-size:1.05rem; margin:8px 0;'>Next review recommended in <b>{lead_time} days</b>.</p>
        </div>
        """, unsafe_allow_html=True)
 
st.divider()
 
# ─────────────────────────────────────────
# INVENTORY POLICY PARAMETERS
# ─────────────────────────────────────────
st.markdown("## Inventory Policy Parameters")
st.markdown(f"<p style='color:#b0d4e8; font-size:1.05rem;'>Active scenario: <b>{selected_scenario}</b> &nbsp;|&nbsp; Unit value: <b>${unit_value}</b> &nbsp;|&nbsp; Lead time: <b>{lead_time} days</b> &nbsp;|&nbsp; K: <b>${K}</b> &nbsp;|&nbsp; h: <b>{h_pct}%</b></p>", unsafe_allow_html=True)
 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Safety Stock", f"{int(selected_policy['SS']):,} units")
col2.metric("Reorder Point", f"{int(selected_policy['ROP']):,} units")
col3.metric("Order Quantity (EOQ)", f"{int(selected_policy['EOQ']):,} units")
col4.metric("Target CSL", f"{selected_policy['CSL']}%")
col5.metric("Expected Annual Cost", f"${selected_policy['TC']:,.2f}")
 
st.divider()
 
# ─────────────────────────────────────────
# PARETO FRONTIER
# ─────────────────────────────────────────
st.markdown("## Cost vs Service Level Trade-off")
st.markdown("<p style='color:#b0d4e8; font-size:1.05rem;'>Each point on the curve represents a non-dominated inventory policy. Moving right increases service level protection but at a higher annual cost. The three scenario operating points are highlighted.</p>", unsafe_allow_html=True)
 
csl_range = np.arange(0.85, 1.00, 0.005)
pareto_csl = []
pareto_cost = []
 
for csl in csl_range:
    p = compute_policy(csl * 100, lead_time, unit_value, K, h_rate)
    pareto_csl.append(csl * 100)
    pareto_cost.append(p['TC'])
 
fig3, ax3 = make_plot(figsize=(12, 5))
ax3.plot(pareto_csl, pareto_cost, '-', color='#7a9bbf', linewidth=2.5, label='Pareto Frontier (XGBoost)')
 
for (scenario, percentile), color in zip(SCENARIO_PERCENTILES.items(),
                                          ['#2ecc71', '#5ba8f5', '#e74c3c']):
    p = compute_policy(percentile, lead_time, unit_value, K, h_rate)
    ax3.scatter(p['CSL'], p['TC'], color=color, s=180, zorder=5,
                edgecolors='white', linewidth=1.5, label=scenario)
    ax3.annotate(f"  {scenario.split('(')[1].replace(')', '')}",
                 (p['CSL'], p['TC']), fontsize=10, color=color, fontweight='bold')
 
ax3.set_xlabel('Target Cycle Service Level (%)', color='#b0d4e8', fontsize=12)
ax3.set_ylabel('Expected Annual Inventory Cost ($)', color='#b0d4e8', fontsize=12)
ax3.set_title('Cost vs Service Level Trade-off — XGBoost Inventory Policy',
              color='#ffffff', fontsize=14, fontweight='bold', pad=15)
ax3.legend(facecolor='#243555', edgecolor='#3a5a80', labelcolor='#e8edf2', fontsize=11)
plt.tight_layout()
st.pyplot(fig3)
 
st.markdown("""
<div style='background-color:#243555; border:1px solid #3a5a80; border-radius:12px; padding:20px; margin-top:16px;'>
    <p style='color:#7ec8e3; font-weight:700; font-size:1.05rem; margin-bottom:10px;'>How to use this chart:</p>
    <p style='color:#d0dce8; font-size:1.0rem; margin:6px 0;'>🟢 <b>Move left</b> — lower cost, lower protection → choose <b>Optimistic</b> scenario</p>
    <p style='color:#d0dce8; font-size:1.0rem; margin:6px 0;'>🔵 <b>Stay at middle</b> — balanced cost and service → choose <b>Base Case</b> scenario</p>
    <p style='color:#d0dce8; font-size:1.0rem; margin:6px 0;'>🔴 <b>Move right</b> — higher cost, maximum protection → choose <b>Pessimistic</b> scenario</p>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: #4a6a8a; font-size: 0.9rem; padding: 10px 0;'>
    Favorita Inventory Decision Tool &nbsp;|&nbsp; Personal Care Category — Store 44, Quito &nbsp;|&nbsp; Corporación Favorita<br>
    Powered by XGBoost &nbsp;·&nbsp; Continuous Review (s,Q) Policy with EOQ &nbsp;·&nbsp; Proof-of-concept tool
</div>
""", unsafe_allow_html=True)
