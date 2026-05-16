import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import timedelta

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Decision Support Tool",
    page_icon="📦",
    layout="wide"
)

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

# Scenario percentiles
SCENARIO_PERCENTILES = {
    'Optimistic (90% CSL)': 90,
    'Base Case (95% CSL)': 95,
    'Pessimistic (99% CSL)': 99
}

SCENARIO_COLORS = {
    'Optimistic (90% CSL)': '#2ecc71',
    'Base Case (95% CSL)': '#3498db',
    'Pessimistic (99% CSL)': '#e74c3c'
}

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def generate_forecast(df, n_days=7):
    """Generate n_days ahead forecast using XGBoost"""
    last_sales = list(df['sales'].values)
    last_row = df.iloc[-1]

    forecast_dates = []
    forecasts = []

    for i in range(1, n_days + 1):
        next_date = LAST_DATE + timedelta(days=i)
        day_of_week = next_date.weekday()
        month = next_date.month
        year = next_date.year
        is_payday = 1 if next_date.day in [15, next_date.replace(
            day=1, month=next_date.month % 12 + 1).replace(day=1) -
            timedelta(days=1)).day] else 0

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

def compute_policy(percentile, L, p):
    h = 0.25 * p
    K = 50
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

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📦 Inventory Decision Support Tool")
st.markdown("**Store 44 — Personal Care Category | Corporación Favorita**")
st.markdown("*Powered by XGBoost demand forecasting — simulating decisions as of August 15, 2017*")
st.divider()

# ─────────────────────────────────────────
# SECTION 1 — MANAGER INPUT
# ─────────────────────────────────────────
st.header("Section 1 — Manager Input")

col1, col2, col3 = st.columns(3)
with col1:
    current_inventory = st.number_input(
        "Current Physical Inventory (units)",
        min_value=0,
        value=5000,
        step=100,
        help="Enter the current number of units on hand"
    )
with col2:
    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=1,
        max_value=21,
        value=7,
        help="Number of days between placing and receiving an order"
    )
with col3:
    unit_value = st.selectbox(
        "Unit Value Scenario",
        options=[2, 4, 8],
        index=1,
        format_func=lambda x: f"${x} per unit",
        help="Assumed average unit value for cost calculations"
    )

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
# SECTION 2 — 7-DAY DEMAND FORECAST
# ─────────────────────────────────────────
st.header("Section 2 — 7-Day Demand Forecast")
st.markdown("*Forward-looking demand forecast generated by the XGBoost model using the last available historical data.*")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(forecast_dates, forecasts, 'o-', color='#3498db',
         linewidth=2, markersize=6, label='Point Forecast')
ax1.fill_between(forecast_dates, lower_bound, upper_bound,
                 alpha=0.2, color='#3498db', label='Uncertainty Band (5th–95th percentile)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Forecasted Units Sold')
ax1.set_title('7-Day Demand Forecast — Personal Care, Store 44')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

forecast_df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
    'Point Forecast (units)': [round(f, 0) for f in forecasts],
    'Lower Bound (units)': [round(l, 0) for l in lower_bound],
    'Upper Bound (units)': [round(u, 0) for u in upper_bound]
})
st.dataframe(forecast_df, use_container_width=True)
st.divider()

# ─────────────────────────────────────────
# SECTION 3 — SCENARIO ANALYSIS
# ─────────────────────────────────────────
st.header("Section 3 — Scenario Analysis")
st.markdown("""
Three inventory scenarios are presented simultaneously, each reflecting a different 
level of demand uncertainty and risk tolerance:

- 🟢 **Optimistic (90% CSL)** — stable demand conditions, no upcoming demand events
- 🔵 **Base Case (95% CSL)** — standard operating conditions
- 🔴 **Pessimistic (99% CSL)** — elevated demand expected (holiday, payday, promotion)
""")

# Compute policies for all three scenarios
policies = {}
for scenario, percentile in SCENARIO_PERCENTILES.items():
    policies[scenario] = compute_policy(percentile, lead_time, unit_value)

# Scenario comparison table
scenario_table = pd.DataFrame({
    'Scenario': list(policies.keys()),
    'Target CSL (%)': [p['CSL'] for p in policies.values()],
    'Safety Stock (units)': [p['SS'] for p in policies.values()],
    'Reorder Point (units)': [p['ROP'] for p in policies.values()],
    'Order Quantity (units)': [p['EOQ'] for p in policies.values()],
    'Expected Annual Cost ($)': [p['TC'] for p in policies.values()],
    'Order Decision': [
        '🔴 Place Order' if current_inventory <= p['ROP'] else '🟢 No Order Needed'
        for p in policies.values()
    ]
})
st.dataframe(scenario_table, use_container_width=True)

# Inventory trajectory plot
st.subheader("Projected Inventory Trajectory — Next 7 Days")
fig2, ax2 = plt.subplots(figsize=(12, 4))

all_dates = [LAST_DATE] + forecast_dates
for scenario, policy in policies.items():
    trajectory = project_inventory(
        current_inventory, forecasts,
        policy['ROP'], policy['EOQ'], lead_time
    )
    ax2.plot(all_dates, trajectory,
             'o-', color=SCENARIO_COLORS[scenario],
             linewidth=2, markersize=5, label=scenario)

ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, label='Zero Inventory')
ax2.set_xlabel('Date')
ax2.set_ylabel('Inventory Level (units)')
ax2.set_title('Projected Inventory Trajectory Under Three Scenarios')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
st.divider()

# ─────────────────────────────────────────
# SECTION 4 — REPLENISHMENT DECISION
# ─────────────────────────────────────────
st.header("Section 4 — Replenishment Decision")

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
    delta = current_inventory - ROP
    st.metric("Buffer above ROP", f"{int(delta):,} units",
              delta=int(delta), delta_color="normal")

if current_inventory <= ROP:
    st.error(f"""
    ### 🔴 PLACE ORDER
    **Order Quantity:** {int(EOQ):,} units  
    **Expected Delivery:** {(LAST_DATE + timedelta(days=lead_time)).strftime('%Y-%m-%d')} ({lead_time} days)  
    **Projected inventory at delivery:** {max(0, int(current_inventory - sum(forecasts[:lead_time]))):,} units
    """)
else:
    days_until_stockout = None
    inv = current_inventory
    for i, demand in enumerate(forecasts):
        inv -= demand
        if inv <= ROP:
            days_until_stockout = i + 1
            break

    if days_until_stockout:
        st.warning(f"""
        ### 🟡 NO ORDER NEEDED — MONITOR CLOSELY
        Inventory is above the reorder point but may reach it within **{days_until_stockout} days**.  
        Consider placing an order soon under the **Pessimistic** scenario.
        """)
    else:
        st.success(f"""
        ### 🟢 NO ORDER NEEDED
        Current inventory of **{current_inventory:,} units** is comfortably above the reorder point of **{int(ROP):,} units**.  
        Next review recommended in **{lead_time} days**.
        """)

st.divider()

# ─────────────────────────────────────────
# SECTION 5 — INVENTORY POLICY PARAMETERS
# ─────────────────────────────────────────
st.header("Section 5 — Inventory Policy Parameters")
st.markdown(f"*Active scenario: **{selected_scenario}** | Unit value: **${unit_value}** | Lead time: **{lead_time} days***")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Safety Stock", f"{int(selected_policy['SS']):,} units")
col2.metric("Reorder Point", f"{int(selected_policy['ROP']):,} units")
col3.metric("Order Quantity (EOQ)", f"{int(selected_policy['EOQ']):,} units")
col4.metric("Target CSL", f"{selected_policy['CSL']}%")
col5.metric("Expected Annual Cost", f"${selected_policy['TC']:,.2f}")

st.divider()

# ─────────────────────────────────────────
# SECTION 6 — PARETO FRONTIER
# ─────────────────────────────────────────
st.header("Section 6 — Cost vs Service Level Trade-off (Pareto Frontier)")
st.markdown("""
The curve below shows the cost-service level trade-off for the XGBoost-based inventory policy. 
Each point on the curve represents a non-dominated solution — moving right increases service level 
but at a higher cost. The three scenario operating points are highlighted.
""")

# Generate Pareto frontier
csl_range = np.arange(0.85, 1.00, 0.005)
pareto_csl = []
pareto_cost = []

for csl in csl_range:
    p = compute_policy(csl * 100, lead_time, unit_value)
    pareto_csl.append(csl * 100)
    pareto_cost.append(p['TC'])

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(pareto_csl, pareto_cost, '-', color='#95a5a6',
         linewidth=2, label='Pareto Frontier (XGBoost)')

# Plot three scenario points
scenario_point_colors = ['#2ecc71', '#3498db', '#e74c3c']
for (scenario, percentile), color in zip(SCENARIO_PERCENTILES.items(),
                                          scenario_point_colors):
    p = compute_policy(percentile, lead_time, unit_value)
    ax3.scatter(p['CSL'], p['TC'], color=color, s=120, zorder=5,
                label=scenario)
    ax3.annotate(f"  {scenario.split('(')[1].replace(')', '')}",
                 (p['CSL'], p['TC']), fontsize=9)

ax3.set_xlabel('Target Cycle Service Level (%)')
ax3.set_ylabel('Expected Annual Inventory Cost ($)')
ax3.set_title('Cost vs Service Level Trade-off — XGBoost Inventory Policy')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

st.markdown("""
**How to use this chart:**
- Move **left** on the curve → lower cost, lower service level protection → choose **Optimistic** scenario
- Stay at the **middle** → balanced cost and service → choose **Base Case** scenario  
- Move **right** on the curve → higher cost, higher protection → choose **Pessimistic** scenario
""")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align: center; color: grey; font-size: 12px;'>
Inventory Decision Support Tool | Personal Care Category — Store 44 | Corporación Favorita<br>
Powered by XGBoost | Framework: Continuous Review (s,Q) Policy with EOQ<br>
Proof-of-concept tool — simulating decisions as of August 15, 2017
</div>
""", unsafe_allow_html=True)