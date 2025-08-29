import streamlit as st
import pandas as pd
import datetime as dt
import math
import yfinance as yf
import numpy as np
from typing import Optional, List, Dict

st.set_page_config(page_title="S&P 500 Short Squeeze Scanner", layout="wide")
st.title("📈 S&P 500 Short Squeeze Scanner")
st.markdown("This app scans S&P 500 stocks for potential short squeeze candidates based on short interest, borrow fees, liquidity, and momentum.")

# -----------------------------
# Core Scanner Logic
# -----------------------------

DEFAULT_LOOKBACK = 90
DEFAULT_ADV_WINDOW = 30
WEIGHTS = {"dtc":0.35, "sipf":0.30, "borrow":0.15, "momentum":0.10, "liquidity":0.05, "volatility":0.05}
CAPS = {"dtc":10.0, "sipf":0.35, "borrow":0.10, "volatility":1.50}

class Metrics:
    def __init__(self, ticker):
        self.ticker = ticker
        self.last_price = None
        self.adv = None
        self.dtc = None
        self.sipf = None
        self.borrow_fee = None
        self.vol90_annual = None
        self.mom_20d = None
        self.mom_5d = None
        self.score = None

# -----------------------------
# Helper Functions
# -----------------------------

def safe_pct(a: float, b: float) -> Optional[float]:
    try:
        return float(a)/float(b) if b else None
    except:
        return None

def compute_adv(df: pd.DataFrame, window:int) -> Optional[float]:
    try:
        return float(df['Volume'].tail(window).mean())
    except:
        return None

def compute_volatility(df: pd.DataFrame, lookback:int=90) -> Optional[float]:
    try:
        px = df['Adj Close'].tail(lookback)
        rets = px.pct_change().dropna()
        return float(rets.std()*math.sqrt(252))
    except:
        return None

def compute_momentum(df: pd.DataFrame, days:int) -> Optional[float]:
    try:
        px = df['Adj Close'].tail(days)
        return float(px.iloc[-1]/px.iloc[0]-1.0)
    except:
        return None

def normalize(value: Optional[float], cap: float) -> Optional[float]:
    if value is None: return None
    return max(0.0,min(float(value),cap))/cap

def compute_score(m: Metrics) -> float:
    n_dtc = normalize(m.dtc, CAPS['dtc'])
    n_sipf = normalize(m.sipf, CAPS['sipf'])
    n_borrow = normalize(m.borrow_fee, CAPS['borrow'])
    mom_score = ((-min(0.0,m.mom_20d)*0.6 + max(0.0,m.mom_5d)*0.4)/0.3) if m.mom_20d is not None and m.mom_5d is not None else None
    liq_score = 1.0 if m.adv and 5e5<=m.adv<=2e7 else 0.0
    vol_score = normalize(m.vol90_annual, CAPS['volatility'])
    parts,weights=[],[]
    for val,w in zip([n_dtc,n_sipf,n_borrow,mom_score,liq_score,vol_score],WEIGHTS.values()):
        if val is not None:
            parts.append(val)
            weights.append(w)
    return float(np.dot(parts,weights)/sum(weights)) if parts else 0.0

# -----------------------------
# Scan Function
# -----------------------------

def run_scan(lookback:int=DEFAULT_LOOKBACK, adv_window:int=DEFAULT_ADV_WINDOW, min_price:float=3, min_adv:float=5e5, top:int=20) -> pd.DataFrame:
    # Get S&P 500 tickers
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        universe = tables[0]['Symbol'].str.upper().tolist()
    except:
        st.error("Failed to fetch S&P 500 list.")
        return pd.DataFrame()

    rows = []
    end = dt.date.today()
    start = end - dt.timedelta(days=max(lookback,adv_window)+10)

    for t in universe:
        try:
            df = yf.download(t,start=start,end=end,progress=False)
            if df.empty: continue
            m = Metrics(t)
            m.last_price = float(df['Adj Close'].iloc[-1])
            m.adv = compute_adv(df,adv_window)
            m.vol90_annual = compute_volatility(df,lookback)
            m.mom_20d = compute_momentum(df,20)
            m.mom_5d = compute_momentum(df,5)
            # Placeholders for optional data
            m.dtc = None
            m.sipf = None
            m.borrow_fee = None
            m.score = compute_score(m)
            if m.last_price>=min_price and m.adv>=min_adv:
                rows.append(m)
        except:
            continue

    out_df = pd.DataFrame([{ 'Ticker':m.ticker, 'Score':round(m.score,4), 'LastPrice':m.last_price, 'ADV':m.adv } for m in rows])
    out_df = out_df.sort_values('Score',ascending=False)
    out_df.to_csv('squeeze_candidates.csv',index=False)
    return out_df.head(top)

# -----------------------------
# Streamlit UI
# -----------------------------

lookback = st.slider("Price history lookback (days)",30,180,90)
adv_window = st.slider("ADV window (days)",10,60,30)
min_price = st.number_input("Minimum stock price ($)",1.0,50.0,3.0)
min_adv = st.number_input("Minimum ADV (shares)",100000,5000000,500000)
top_n = st.slider("Number of top candidates to display",5,50,20)

if st.button("🔍 Run Scan"):
    with st.spinner("Scanning S&P 500..."):
        df = run_scan(lookback,adv_window,min_price,min_adv,top_n)
    st.success("Scan completed!")
    if not df.empty:
        st.dataframe(df)
        st.download_button("💾 Download CSV", df.to_csv(index=False), file_name='squeeze_candidates.csv',mime='text/csv')
    else:
        st.warning("No candidates found.")
