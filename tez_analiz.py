import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_to_excel(df, filename):
    full_path = os.path.join(SCRIPT_DIR, filename)
    try:
        df.to_excel(full_path)
        print(f"   [+] Table Saved: {filename}")
    except Exception as e:
        print(f"   [-] Save Error: {e}")

def save_plot(filename):
    full_path = os.path.join(SCRIPT_DIR, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"   [+] Plot Saved: {filename}")
    plt.close() # Close memory to avoid overlapping plots

# =============================================================================
# 1. DATA LOADING & TRANSLATION
# =============================================================================
def load_data(file_path):
    print(f"\n--- 1. LOADING DATA ---")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
        
    df.columns = [col.strip() for col in df.columns]
    
    # Translate Columns (Turkish -> English)
    column_mapping = {
        'Kredi_Hacmi': 'Credit_Volume',
        'Faiz_Orani': 'Interest_Rate',
        'Tuketici_Guveni': 'Consumer_Confidence',
        'USD_TRY': 'Exchange_Rate',
        'TUFE': 'CPI'  # Consumer Price Index
    }
    df.rename(columns=column_mapping, inplace=True)
    
    df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True)
    df.set_index('Tarih', inplace=True)
    
    if df.isnull().values.any():
        df = df.interpolate(method='linear')
        
    print(f"Data Loaded: {len(df)} monthly observations from {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    return df

# =============================================================================
# 2. VISUALIZATION (Time Series & Correlation)
# =============================================================================
def visualize_data(df):
    print(f"\n--- 2. GENERATING VISUALIZATIONS ---")
    
    # A) General Time Series Plot
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(len(df.columns), 1, i+1)
        plt.plot(df.index, df[col], label=col, color='#1f77b4', linewidth=2)
        plt.title(f"{col} - Historical Trend", fontsize=10)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot('1_general_time_series.png')
    
    # B) Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Macroeconomic Correlation Matrix', fontsize=14)
    save_plot('2_correlation_matrix.png')

# =============================================================================
# 3. MODELING & VALIDATION & FORECASTING
# =============================================================================
def run_analysis(df, target_col, exog_cols):
    print("\n--- 3. TRAINING ARIMAX MODEL (Full Dataset) ---")
    
    # Log Transformation
    df['Log_Target'] = np.log(df[target_col])
    
    # Train Auto ARIMA
    print("Optimizing model parameters (this may take a moment)...")
    model = auto_arima(
        y=df['Log_Target'],
        X=df[exog_cols], 
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None, seasonal=False,
        stepwise=True, suppress_warnings=True, trace=False
    )
    
    print("\nBest Model Selected:")
    print(model.summary())

    # --- C) MODEL VALIDATION PLOT (Actual vs Fitted) ---
    print("\nGenerating Model Validation Plot...")
    # Get in-sample predictions (fitted values)
    fitted_log = model.predict_in_sample(X=df[exog_cols])
    fitted_values = np.exp(fitted_log) # Inverse Log
    
    # Conversion to Billion TL
    DIVISOR = 1_000_000 
    actual_billion = df[target_col] / DIVISOR
    fitted_billion = fitted_values / DIVISOR
    
    plt.figure(figsize=(14, 7))
    plt.plot(actual_billion.index, actual_billion, label='Actual Data', color='black', linewidth=2)
    plt.plot(fitted_billion.index, fitted_billion, label='Model Fit (ARIMAX)', color='red', linestyle='--', alpha=0.8)
    plt.title(f'Model Validation: Actual vs Fitted Values ({target_col})', fontsize=14)
    plt.ylabel('Credit Volume (Billion TL)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    save_plot('3_model_validation_plot.png')

    # --- SCENARIO GENERATION (2025 Forecast) ---
    print("\n--- 4. FORECASTING 2025 SCENARIOS ---")
    
    last_row = df.iloc[-1]
    periods = 6
    future_dates = pd.date_range(start='2025-01-01', periods=periods, freq='MS')
    
    # Worst Case (Stress)
    worst_data = []
    vals = last_row.copy()
    for _ in range(periods):
        vals['Interest_Rate'] *= 1.05       
        vals['Exchange_Rate'] *= 1.03       
        vals['Consumer_Confidence'] *= 0.95 
        vals['CPI'] *= 1.04                 
        worst_data.append(vals[exog_cols].values)
    df_worst = pd.DataFrame(worst_data, columns=exog_cols, index=future_dates)
    
    # Best Case (Stability)
    best_data = []
    vals = last_row.copy()
    for _ in range(periods):
        vals['Interest_Rate'] *= 0.98       
        vals['Exchange_Rate'] *= 1.01       
        vals['Consumer_Confidence'] *= 1.02 
        vals['CPI'] *= 1.015                
        best_data.append(vals[exog_cols].values)
    df_best = pd.DataFrame(best_data, columns=exog_cols, index=future_dates)

    # Predict
    pred_log_worst = model.predict(n_periods=periods, X=df_worst)
    pred_log_best = model.predict(n_periods=periods, X=df_best)
    
    pred_worst = np.exp(pred_log_worst)
    pred_best = np.exp(pred_log_best)
    
    # Combine & Export
    baseline = pd.DataFrame({'Worst Case': [df[target_col].iloc[-1]], 'Best Case': [df[target_col].iloc[-1]]}, index=[df.index[-1]])
    forecast_df = pd.DataFrame({'Worst Case': pred_worst.values, 'Best Case': pred_best.values}, index=future_dates)
    full_df = pd.concat([baseline, forecast_df])
    
    full_df_billion = full_df / DIVISOR
    full_df_billion['Risk Gap'] = full_df_billion['Best Case'] - full_df_billion['Worst Case']
    
    print("\n--- 2025 FORECAST TABLE (Billion TL) ---")
    pd.options.display.float_format = '{:,.2f}'.format
    print(full_df_billion)
    save_to_excel(full_df_billion, '2025_forecast_results.xlsx')
    
    # --- D) FORECAST PLOT ---
    plt.figure(figsize=(12, 7))
    last_12 = actual_billion.iloc[-12:]
    plt.plot(last_12.index, last_12, color='black', label='Actual History (2024)', linewidth=2, marker='o')
    plt.plot(full_df_billion.index, full_df_billion['Best Case'], 'g--o', label='Best Case (Stability)', linewidth=2)
    plt.plot(full_df_billion.index, full_df_billion['Worst Case'], 'r--x', label='Worst Case (Stress)', linewidth=2)
    plt.fill_between(full_df_billion.index, full_df_billion['Best Case'], full_df_billion['Worst Case'], color='gray', alpha=0.15)
    
    plt.title('Consumer Credit Volume Forecast: 2025 Scenarios', fontsize=14)
    plt.ylabel('Credit Volume (Billion TL)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annotations
    end_best = full_df_billion['Best Case'].iloc[-1]
    end_worst = full_df_billion['Worst Case'].iloc[-1]
    plt.annotate(f"{end_best:,.0f} B", (full_df_billion.index[-1], end_best), xytext=(10,5), textcoords='offset points', color='green', fontweight='bold')
    plt.annotate(f"{end_worst:,.0f} B", (full_df_billion.index[-1], end_worst), xytext=(10,-15), textcoords='offset points', color='red', fontweight='bold')
    
    save_plot('4_forecast_scenarios_2025.png')
    
    # Executive Report
    print(f"\n=== EXECUTIVE SUMMARY ===")
    print(f"Total Liquidity Risk (6 Months): {full_df_billion['Risk Gap'].sum():,.2f} Billion TL")
    print("All plots and tables have been saved to the project folder.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    filename = 'manuel_veri.xlsx' 
    file_path = os.path.join(SCRIPT_DIR, filename)
    
    # Check if file exists, if not check for csv
    if not os.path.exists(file_path):
        filename = 'manuel_veri.xlsx - EVDS.csv'
        file_path = os.path.join(SCRIPT_DIR, filename)
    
    if os.path.exists(file_path):
        # 1. Load
        df = load_data(file_path)
        
        # 2. Visualize
        visualize_data(df)
        
        # 3. Analyze
        target = 'Credit_Volume'
        exog = ['Interest_Rate', 'Consumer_Confidence', 'Exchange_Rate', 'CPI']
        run_analysis(df, target, exog)
        
        print("\n[SUCCESS] Analysis Complete.")
    else:
        print(f"[ERROR] File not found: {filename}")