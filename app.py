# Guncelleme kontrol versiyon 2
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="Consumer Credit DSS",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# --- CSS İLE GÖRSELLİĞİ ARTIRMA (BÜYÜK VE KALIN SEKMELER) ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* --- 1. SIDEBAR (YAN MENÜ) GENİŞLİĞİ --- */
   
    [data-testid="stSidebar"] {
        min-width: 600px !important;
        max-width: 800px !important;
    }
    
    /* --- 2. SEKMELERİN GÖRÜNÜMÜ --- */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        padding-bottom: 10px;
        border-bottom: 3px solid #E0E0E0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 10px;
        color: #555555;
        font-weight: 700;
        font-size: 20px;
        border: 3px solid #CCCCCC;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
        flex-grow: 1;
        text-align: center;
        transition: all 0.2s ease-in-out;
        display: flex;
        align_items: center;
        justify_content: center;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: 3px solid #FF4B4B; 
        font-weight: bold;
        transform: scale(1.02);
        box-shadow: 0px 4px 8px rgba(255, 75, 75, 0.4);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #FFE5E5;
        color: #FF4B4B;
        border-color: #FF4B4B;
        cursor: pointer;
    }
    
    /* --- MOBİL İÇİN ÖZEL AYARLAR --- */
    @media only screen and (max-width: 600px) {
        .stTabs [data-baseweb="tab"] {
            font-size: 12px !important;
            height: 40px !important;
            padding: 2px !important;
        }
        h1 { font-size: 24px !important; }
        
        /* Mobilde sidebar otomatik daralsın */
        [data-testid="stSidebar"] {
            min-width: 100% !important;
        }
    }
    
    /* İstatistik Kartları */
    .stat-card {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #212529;
    }
    .stat-label {
        font-size: 14px;
        color: #6C757D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metrik Kutuları */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- 1. VERİ YÜKLEME ---
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = 'manuel_veri.xlsx'
    path = os.path.join(script_dir, filename)
    
    if not os.path.exists(path):
        path = os.path.join(script_dir, 'manuel_veri.xlsx - EVDS.csv')
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    else:
        df = pd.read_excel(path)

    df.columns = [col.strip() for col in df.columns]
    column_mapping = {
        'Kredi_Hacmi': 'Credit_Volume',
        'Faiz_Orani': 'Interest_Rate',
        'Tuketici_Guveni': 'Consumer_Confidence',
        'USD_TRY': 'Exchange_Rate',
        'TUFE': 'CPI'
    }
    df.rename(columns=column_mapping, inplace=True)
    df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True)
    df.set_index('Tarih', inplace=True)
    
    if df.isnull().values.any():
        df = df.interpolate(method='linear')
    return df

@st.cache_resource
def train_model(df):
    target_col = 'Credit_Volume'
    exog_cols = ['Interest_Rate', 'Consumer_Confidence', 'Exchange_Rate', 'CPI']
    df['Log_Target'] = np.log(df[target_col])
    
    model = auto_arima(
        y=df['Log_Target'],
        X=df[exog_cols], 
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None, seasonal=False,
        stepwise=True, suppress_warnings=True
    )
    return model

try:
    df = load_data()
    model = train_model(df)
except Exception as e:
    st.error(f"Data Error: {e}")
    st.stop()

# --- SIDEBAR ---

with st.sidebar:
    st.markdown("## 🎛️ Forecast Simulator")
    st.markdown("Adjust the drivers below to simulate **H1 2025** scenarios.")
    st.divider()

    # 1. BAŞLANGIÇ DEĞERLERİ (INITIALIZATION)
    # Sayfa ilk açıldığında hafızada bu değerler yoksa oluşturuyoruz
    if "interest_slider" not in st.session_state: st.session_state.interest_slider = 0.0
    if "interest_input" not in st.session_state: st.session_state.interest_input = 0.0
    
    if "exchange_slider" not in st.session_state: st.session_state.exchange_slider = 2.0
    if "exchange_input" not in st.session_state: st.session_state.exchange_input = 2.0
    
    if "conf_slider" not in st.session_state: st.session_state.conf_slider = 0.0
    if "conf_input" not in st.session_state: st.session_state.conf_input = 0.0
    
    if "cpi_slider" not in st.session_state: st.session_state.cpi_slider = 3.0
    if "cpi_input" not in st.session_state: st.session_state.cpi_input = 3.0

    # 2. SENKRONİZASYON FONKSİYONLARI (CALLBACKS)
    # Biri değiştiğinde DİĞERİNİN hafızasını güncelliyoruz
    def update_interest(source):
        if source == 'slider':
            st.session_state.interest_input = st.session_state.interest_slider
        elif source == 'input':
            st.session_state.interest_slider = st.session_state.interest_input
            
    def update_exchange(source):
        if source == 'slider':
            st.session_state.exchange_input = st.session_state.exchange_slider
        elif source == 'input':
            st.session_state.exchange_slider = st.session_state.exchange_input

    def update_conf(source):
        if source == 'slider':
            st.session_state.conf_input = st.session_state.conf_slider
        elif source == 'input':
            st.session_state.conf_slider = st.session_state.conf_input

    def update_cpi(source):
        if source == 'slider':
            st.session_state.cpi_input = st.session_state.cpi_slider
        elif source == 'input':
            st.session_state.cpi_slider = st.session_state.cpi_input

    # 3. RESET FONKSİYONU
    def reset_sliders():
        # Tüm araçların hafızasını sıfırlıyoruz
        st.session_state.interest_slider = 0.0
        st.session_state.interest_input = 0.0
        
        st.session_state.exchange_slider = 0.0
        st.session_state.exchange_input = 0.0
        
        st.session_state.conf_slider = 0.0
        st.session_state.conf_input = 0.0
        
        st.session_state.cpi_slider = 0.0
        st.session_state.cpi_input = 0.0

    # --- ARAYÜZ ELEMANLARI ---
    
    # 1. Interest Rate
    st.markdown("### 🏦 Monetary Policy")
    col1, col2 = st.columns([3, 1])
    with col1:
        # Slider değişince 'update_interest' çalışır ve input'u günceller
        interest_change = st.slider(
            "Interest Rate (%)", -10.0, 10.0, step=0.1, format="%+.1f%%",
            key="interest_slider", on_change=update_interest, args=('slider',)
        )
    with col2:
        # Input değişince 'update_interest' çalışır ve slider'ı günceller
        st.number_input(
            "Manual", -10.0, 10.0, step=0.1,
            key="interest_input", on_change=update_interest, args=('input',), 
            label_visibility="collapsed"
        )
    
    # 2. Exchange Rate
    st.markdown("### 💲 Currency Market")
    col3, col4 = st.columns([3, 1])
    with col3:
        exchange_change = st.slider(
            "USD/TRY (%)", -10.0, 10.0, step=0.1, format="%+.1f%%",
            key="exchange_slider", on_change=update_exchange, args=('slider',)
        )
    with col4:
        st.number_input(
            "Manual", -10.0, 10.0, step=0.1,
            key="exchange_input", on_change=update_exchange, args=('input',), 
            label_visibility="collapsed"
        )
    
    # 3. Consumer Confidence
    st.markdown("### 📊 Macro Indicators")
    col5, col6 = st.columns([3, 1])
    with col5:
        conf_change = st.slider(
            "Confidence (%)", -10.0, 10.0, step=0.1, format="%+.1f%%",
            key="conf_slider", on_change=update_conf, args=('slider',)
        )
    with col6:
        st.number_input(
            "Manual", -10.0, 10.0, step=0.1,
            key="conf_input", on_change=update_conf, args=('input',), 
            label_visibility="collapsed"
        )

    # 4. CPI
    col7, col8 = st.columns([3, 1])
    with col7:
        cpi_change = st.slider(
            "Inflation / CPI (%)", -10.0, 10.0, step=0.1, format="%+.1f%%",
            key="cpi_slider", on_change=update_cpi, args=('slider',)
        )
    with col8:
        st.number_input(
            "Manual", -10.0, 10.0, step=0.1,
            key="cpi_input", on_change=update_cpi, args=('input',), 
            label_visibility="collapsed"
        )
    
    st.divider()
    st.button("↺ Reset Parameters", on_click=reset_sliders)
    st.info("Values represent monthly rate of change.")

# --- ANA EKRAN ---
st.title("📊 Consumer Credit Volume Decision Support System")

st.markdown("""
This application uses **ARIMAX** models to forecast **Consumer Credit Volume (Household Liquidity)** for the upcoming 6 months based on macroeconomic variables. 
Use the sidebar to adjust simulation parameters and perform **"What-If"** analysis.
""")

st.write("")

# İSTEK: BÜYÜK VE KALIN ÇERÇEVELİ SEKMELER
tab1, tab2, tab3 = st.tabs(["🚀 Forecast Simulator", "📉 Historical Trends", "🧠 Model Insights"])

# =============================================================================
# TAB 1: SIMULATOR
# =============================================================================
with tab1:
    st.markdown("### 🎯 Real-Time Scenario Analysis")
    
    last_row = df.iloc[-1]
    periods = 6
    future_dates = pd.date_range(start='2025-01-01', periods=periods, freq='MS')
    exog_cols = ['Interest_Rate', 'Consumer_Confidence', 'Exchange_Rate', 'CPI']

    # User Scenario
    user_data = []
    base_data = []
    vals_user = last_row.copy()
    vals_base = last_row.copy()
    
    for _ in range(periods):
        vals_user['Interest_Rate'] *= (1 + interest_change/100)
        vals_user['Exchange_Rate'] *= (1 + exchange_change/100)
        vals_user['Consumer_Confidence'] *= (1 + conf_change/100)
        vals_user['CPI'] *= (1 + cpi_change/100)
        user_data.append(vals_user[exog_cols].values)
        
        vals_base['Interest_Rate'] *= 1.00 
        vals_base['Exchange_Rate'] *= 1.01 
        vals_base['Consumer_Confidence'] *= 1.00
        vals_base['CPI'] *= 1.02
        base_data.append(vals_base[exog_cols].values)

    pred_log_user = model.predict(n_periods=periods, X=pd.DataFrame(user_data, columns=exog_cols))
    pred_log_base = model.predict(n_periods=periods, X=pd.DataFrame(base_data, columns=exog_cols))
    
    pred_user = np.exp(pred_log_user) / 1_000_000 # Billion TL
    pred_base = np.exp(pred_log_base) / 1_000_000 # Billion TL
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'User Scenario (Billion TL)': pred_user.values,
        'Baseline (Stable) (Billion TL)': pred_base.values
    })
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')

    # KPI
    col1, col2, col3, col4 = st.columns(4)
    last_actual_val = df['Credit_Volume'].iloc[-1] / 1_000_000
    forecast_end_val = pred_user.values[-1]
    growth_rate = ((forecast_end_val - last_actual_val) / last_actual_val) * 100
    gap_vs_baseline = forecast_end_val - pred_base.values[-1]

    col1.metric("Current Volume (Dec 2024)", f"{last_actual_val:,.1f} B TL")
    col2.metric("Forecast (Jun 2025)", f"{forecast_end_val:,.1f} B TL", delta=f"{growth_rate:.1f}% Growth")
    col3.metric("Impact of Your Scenario", f"{gap_vs_baseline:,.1f} B TL", delta_color="normal" if gap_vs_baseline > 0 else "inverse")
    col4.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False).encode('utf-8'), 'forecast_2025.csv', 'text/csv')

    st.markdown("---")

    col_chart, col_insight = st.columns([3, 1])
    with col_chart:
        st.subheader("📊 Scenario Comparison Chart")
        fig = go.Figure()
        past_df = df.iloc[-18:]
        
        conn_x = [past_df.index[-1]] + list(pd.to_datetime(forecast_df['Date']))
        conn_y_base = [past_df['Credit_Volume'].iloc[-1]/1_000_000] + list(pred_base.values)
        conn_y_user = [past_df['Credit_Volume'].iloc[-1]/1_000_000] + list(pred_user.values)

        fig.add_trace(go.Scatter(x=past_df.index, y=past_df['Credit_Volume']/1_000_000, mode='lines+markers', name='Actual History', line=dict(color='#333333', width=3)))
        fig.add_trace(go.Scatter(x=conn_x, y=conn_y_base, mode='lines', name='Baseline', line=dict(color='gray', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=conn_x, y=conn_y_user, mode='lines+markers', name='Your Scenario', line=dict(color='#FF4B4B', width=4)))
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Date", yaxis_title="Volume (Billion TL)", hovermode="x unified", template="plotly_white", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_insight:
        st.subheader("💡 AI Insight")
        if gap_vs_baseline < -50:
            st.error(f"**High Risk**\nGap: **{abs(gap_vs_baseline):.1f} B TL**\nLevel: 🔴 CRITICAL")
        elif gap_vs_baseline > 50:
            st.success(f"**Opportunity**\nGap: **{gap_vs_baseline:.1f} B TL**\nLevel: 🟢 LOW")
        else:
            st.info("**Neutral**\nScenario matches baseline.\nLevel: 🟡 MODERATE")

    st.markdown("---")
    st.subheader("📋 Detailed Forecast Data Table")
    st.dataframe(
        forecast_df.style.format({
            'User Scenario (Billion TL)': "{:,.2f}", 
            'Baseline (Stable) (Billion TL)': "{:,.2f}"
        }), 
        use_container_width=True, 
        height=250
    )

# =============================================================================
# TAB 2: GEÇMİŞ VERİ
# =============================================================================
with tab2:
    st.markdown("### 📈 Historical Trends Analysis")
    
    with st.container():
        col_ctrl, col_stats = st.columns([1, 3])
        with col_ctrl:
            st.markdown("**Select Indicator:**")
            variable = st.selectbox("Choose a variable to visualize:", df.columns, label_visibility="collapsed")
            
        with col_stats:
            stats = df[variable].describe()
            change_pct = ((df[variable].iloc[-1] - df[variable].iloc[0]) / df[variable].iloc[0]) * 100
            
            st.markdown(f"""
            <div style="display: flex; gap: 20px; margin-bottom: 20px;">
                <div class="stat-card" style="flex:1;">
                    <div class="stat-label">Average</div>
                    <div class="stat-value">{stats['mean']:,.2f}</div>
                </div>
                <div class="stat-card" style="flex:1;">
                    <div class="stat-label">Minimum</div>
                    <div class="stat-value">{stats['min']:,.2f}</div>
                </div>
                <div class="stat-card" style="flex:1;">
                    <div class="stat-label">Maximum</div>
                    <div class="stat-value">{stats['max']:,.2f}</div>
                </div>
                <div class="stat-card" style="flex:1; border-left: 5px solid {'green' if change_pct > 0 else 'red'};">
                    <div class="stat-label">Total Change (10Y)</div>
                    <div class="stat-value">{change_pct:+.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_main_chart, col_dist_chart = st.columns([2, 1])
    
    with col_main_chart:
        st.markdown("**📉 Evolution Over Time**")
        fig_line = px.line(df, x=df.index, y=variable, markers=True)
        fig_line.update_traces(line_color='#1f77b4', line_width=2)
        fig_line.update_layout(template="plotly_white", height=400, margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_dist_chart:
        st.markdown("**📊 Data Distribution (Histogram)**")
        fig_hist = px.histogram(df, x=variable, nbins=20, opacity=0.7)
        fig_hist.update_traces(marker_color='#FF4B4B')
        fig_hist.update_layout(template="plotly_white", height=400, margin=dict(l=20,r=20,t=30,b=20), yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        with st.expander("ℹ️ How to interpret this chart?"):
            st.markdown("""
            **What does this histogram show?**
            * **Tall Bars:** Represent the most common range of values (The 'Normal' state).
            * **Wide Spread:** Indicates high volatility and uncertainty.
            * **Isolated Bars:** Outliers representing economic shocks.
            """)

    st.markdown("---")
    st.subheader(f"🔍 Detailed Data View: {variable}")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.write("**Statistical Summary**")
        st.table(pd.DataFrame(stats))
    with col_t2:
        st.write("**Last 12 Months Data**")
        styled_df = df[[variable]].tail(12).style.background_gradient(cmap="Blues")
        st.dataframe(styled_df, use_container_width=True)

# =============================================================================
# TAB 3: MODEL İÇGÖRÜLERİ
# =============================================================================
with tab3:
    st.markdown("### 🧠 Model Performance & Diagnostics")
    
    exog_cols = ['Interest_Rate', 'Consumer_Confidence', 'Exchange_Rate', 'CPI']
    fitted_log = model.predict_in_sample(X=df[exog_cols])
    fitted_values = np.exp(fitted_log)
    actual_values = df['Credit_Volume']
    
    mae = mean_absolute_error(actual_values, fitted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, fitted_values))
    r2 = r2_score(actual_values, fitted_values)
    aic = model.aic()

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("R-Squared (Accuracy)", f"{r2:.2%}")
    col_kpi2.metric("AIC Score", f"{aic:.0f}")
    col_kpi3.metric("MAE (Mean Error)", f"{mae/1_000_000:,.1f} M")
    col_kpi4.metric("RMSE", f"{rmse/1_000_000:,.1f} M")

    st.markdown("---")
    col_fit, col_resid = st.columns([2, 1])

    with col_fit:
        st.subheader("✅ Model Fit: Actual vs Predicted")
        fit_df = pd.DataFrame({'Actual': actual_values / 1_000_000, 'Model Fit': fitted_values / 1_000_000}, index=df.index)
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=fit_df.index, y=fit_df['Actual'], mode='lines', name='Actual Data', line=dict(color='black', width=2)))
        fig_fit.add_trace(go.Scatter(x=fit_df.index, y=fit_df['Model Fit'], mode='lines', name='Model Prediction', line=dict(color='red', dash='dot', width=2)))
        fig_fit.update_layout(height=400, template="plotly_white", hovermode="x unified", yaxis_title="Billion TL", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_fit, use_container_width=True)

    with col_resid:
        st.subheader("🔥 Correlations")
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("🛠️ View Technical Model Parameters (JSON)"):
        st.json({
            "Model Type": "ARIMAX",
            "ARIMA Order": str(model.order),
            "Exogenous Variables": exog_cols,
            "Total Observations": len(df),
            "Transformation": "Logarithmic"
        })