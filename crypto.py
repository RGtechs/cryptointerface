import streamlit as st
import ccxt
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import time
st.query_params["refresh"] = int(time.time())

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Real-time BTC/USDT Dashboard")

exchange = ccxt.binance()

def fetch_all_ohlcv(exchange, symbol, timeframe, since, to):
    all_ohlcv = []
    while since < to:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not data:
                break
            all_ohlcv.extend(data)
            last_timestamp = data[-1][0]
            if last_timestamp == since:
                break  # avoid infinite loop
            since = last_timestamp + 1
            time.sleep(exchange.rateLimit / 1000)  # to respect Binance rate limits
        except Exception as e:
            st.warning(f"⚠️ Error during data fetch: {e}")
            break
    return all_ohlcv


# Load available markets from Binance
markets = exchange.load_markets()
available_symbols = list(markets.keys())

# User input for base and quote currencies
base_currency = st.text_input("Enter Base Currency (e.g., BTC, ETH)", "BTC").upper()
quote_currency = st.text_input("Enter Quote Currency (e.g., USDT, EUR)", "USDT").upper()

# Construct symbol
symbol = f"{base_currency}/{quote_currency}"

if symbol not in available_symbols:
    st.error(f"❌ Trading pair {symbol} not available on Binance.")
    
    # Suggest valid quote currencies for the selected base
    suggested = [s for s in available_symbols if s.startswith(f"{base_currency}/")]
    if suggested:
        st.info("✅ Did you mean one of these?\n\n" + ", ".join(suggested[:10]))
    else:
        st.info("ℹ️ No matching pairs found.")
    
    st.stop()



from datetime import timedelta

timeframes = {
    "1 Day": ("1m", timedelta(days=1)),
    "1 Week": ("5m", timedelta(weeks=1)),
    "1 Month": ("30m", timedelta(days=30)),
    "1 Year": ("1d", timedelta(days=365)),
    "500 Days": ("1d", timedelta(days=500)),
}

selected = st.selectbox("Select Time Range", list(timeframes.keys()), index=0)
tf, delta = timeframes[selected]

now = exchange.milliseconds()
if delta is not None:
    since = now - int(delta.total_seconds() * 1000)
    ohlcv = fetch_all_ohlcv(exchange, symbol, tf, since, now)
else:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=5000)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

if df.empty:
    st.error("⚠️ No data returned for the selected timeframe.")
    st.stop()

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

y_min = 0.99999 * df['low'].min()
y_max = 1.00001 * df['high'].max()

latest = df.iloc[-1]
first = df.iloc[0]
percentage = (latest['close'] - first['close']) * 100 / first['close']

def showpercentage(pct: float) -> str:
    color = "green" if pct > 0 else "red"
    sign = "+" if pct > 0 else ""
    return f"<span style='color:{color}'>{sign}{pct:.2f}%</span>"


fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='BTC Close Price'))

fig.update_layout(
    title={
        "text": f"{base_currency}/{quote_currency} Price ({selected}) {showpercentage(percentage)}",
        "x": 0.5,
        "xanchor": "center"
    },
    yaxis=dict(
        title=f"Price ({quote_currency})",
        range=[y_min, y_max], 
        rangemode='tozero',
        fixedrange=False       
    ),
    xaxis=dict(
        title="Time",
        range=[df['timestamp'].min(), df['timestamp'].max()],
        rangemode='tozero',
        fixedrange=False
    ),
    template='plotly_dark',
    height=400,
    dragmode='zoom', 
    margin=dict(l=0, r=0, t=40, b=0),
    modebar_remove=["zoomIn2d", "zoomOut2d"]
)

st.plotly_chart(fig, use_container_width=True)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.metric(label=f"Price (as of {now})", value=f"{quote_currency} {latest['close']:,.2f}")

vol = sum(df['volume'])
st.metric(label="Volume", value=vol)

from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator
import numpy as np

st.header("Technical Indicators")

# --- OBV ---
obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
fig_obv = go.Figure()
fig_obv.add_trace(go.Scatter(x=df["timestamp"], y=obv, name="OBV", line=dict(color='aqua')))
fig_obv.update_layout(title="On-Balance Volume (OBV)", template="plotly_dark", height=300)
st.plotly_chart(fig_obv, use_container_width=True)

# --- CVD (simplified) ---
df["delta"] = df["close"].diff()
df["buy_volume"] = np.where(df["delta"] > 0, df["volume"], 0)
df["sell_volume"] = np.where(df["delta"] < 0, df["volume"], 0)
df["cvd"] = (df["buy_volume"] - df["sell_volume"]).cumsum()

fig_cvd = go.Figure()
fig_cvd.add_trace(go.Scatter(x=df["timestamp"], y=df["cvd"], name="CVD", line=dict(color='orange')))
fig_cvd.update_layout(title="Cumulative Volume Delta (CVD)", template="plotly_dark", height=300)
st.plotly_chart(fig_cvd, use_container_width=True)

# --- Volume Bar Chart ---
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df["timestamp"], y=df["volume"], marker_color="gray", name="Volume"))
fig_vol.update_layout(title="Volume", template="plotly_dark", height=300)
st.plotly_chart(fig_vol, use_container_width=True)

# --- RSI ---
rsi = RSIIndicator(close=df["close"], window=14).rsi()
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["timestamp"], y=rsi, name="RSI", line=dict(color='lime')))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="top left")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue", annotation_text="Oversold", annotation_position="bottom left")
fig_rsi.update_layout(title="Relative Strength Index (RSI)", template="plotly_dark", height=300)
st.plotly_chart(fig_rsi, use_container_width=True)





import plotly.express as px

# Fetch order book
order_book = exchange.fetch_order_book(symbol)
bids_all = order_book['bids']   # All bids for charts
asks_all = order_book['asks']   # All asks for charts

# For display only (first 100 rows for performance)
bids_display = bids_all[:100]
asks_display = asks_all[:100]

# --- THRESHOLD CONTROLS ---
st.sidebar.header("🔧 Order Size Thresholds")
small_max = st.sidebar.slider("Max size for Small", 0.01, 5.0, 0.5)
medium_max = st.sidebar.slider("Max size for Medium", small_max, 50.0, 5.0)

# --- CATEGORIZATION FUNCTION FOR VOLUME ---
def categorize_orders(orders):
    categories = {"Small": 0, "Medium": 0, "Big": 0}
    for _, amount in orders:
        if amount < small_max:
            categories["Small"] += amount
        elif amount < medium_max:
            categories["Medium"] += amount
        else:
            categories["Big"] += amount
    return categories

# --- CATEGORIZATION FUNCTION FOR COUNT ---
def count_orders(orders):
    count = {"Small": 0, "Medium": 0, "Big": 0}
    for _, amount in orders:
        if amount < small_max:
            count["Small"] += 1
        elif amount < medium_max:
            count["Medium"] += 1
        else:
            count["Big"] += 1
    return count

# Categorize full order book for volume
bid_vol_cat = categorize_orders(bids_all)
ask_vol_cat = categorize_orders(asks_all)

# Categorize full order book for count (for the extra bar chart you want)
bid_count_cat = count_orders(bids_all)
ask_count_cat = count_orders(asks_all)

# --- PIE CHART DATA (volume-based) ---
pie_df = pd.DataFrame({
    "Category": [
        "Small Bids", "Medium Bids", "Big Bids",
        "Small Asks", "Medium Asks", "Big Asks"
    ],
    "Volume": [
        bid_vol_cat["Small"], bid_vol_cat["Medium"], bid_vol_cat["Big"],
        ask_vol_cat["Small"], ask_vol_cat["Medium"], ask_vol_cat["Big"]
    ]
})

fig_pie = px.pie(
    pie_df,
    names="Category",
    values="Volume",
    hole=0.45,
    color="Category",
    color_discrete_map={
        "Small Bids": "#00cc96",
        "Medium Bids": "#f5c518",
        "Big Bids": "#0074D9",
        "Small Asks": "#ff4d4d",
        "Medium Asks": "#FF851B",
        "Big Asks": "#B10DC9"
    }
)

fig_pie.update_traces(
    textinfo="percent",
    hoverinfo="label+value+percent",
    pull=[0.03] * 6,
    sort=False,
    marker=dict(line=dict(color='#1e1e1e', width=2))
)

fig_pie.update_layout(
    title_text="Order Volume Split by Type & Size",
    title_x=0.02,
    showlegend=True,
    height=360,
    legend_title="Category",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5,
        font=dict(size=11)
    ),
    margin=dict(t=50, b=20, l=10, r=10),
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="white"
)

# --- BAR CHART VOLUME ---
bar_vol_df = pd.DataFrame({
    "Size": ["Small", "Medium", "Big"],
    "Bids": [bid_vol_cat["Small"], bid_vol_cat["Medium"], bid_vol_cat["Big"]],
    "Asks": [ask_vol_cat["Small"], ask_vol_cat["Medium"], ask_vol_cat["Big"]]
})

fig_bar_vol = go.Figure(data=[
    go.Bar(name='Bids', x=bar_vol_df["Size"], y=bar_vol_df["Bids"], marker_color="#00cc96"),
    go.Bar(name='Asks', x=bar_vol_df["Size"], y=bar_vol_df["Asks"], marker_color="#ff4d4d")
])
fig_bar_vol.update_layout(
    barmode='group',
    title_text="Order Volume by Size",
    xaxis_title="Order Size",
    yaxis_title="Volume",
    height=320,
    template="plotly_dark",
    margin=dict(t=50, b=20),
)

# --- BAR CHART COUNT ---
bar_count_df = pd.DataFrame({
    "Size": ["Small", "Medium", "Big"],
    "Bids": [bid_count_cat["Small"], bid_count_cat["Medium"], bid_count_cat["Big"]],
    "Asks": [ask_count_cat["Small"], ask_count_cat["Medium"], ask_count_cat["Big"]]
})

fig_bar_count = go.Figure(data=[
    go.Bar(name='Bids', x=bar_count_df["Size"], y=bar_count_df["Bids"], marker_color="#00cc96"),
    go.Bar(name='Asks', x=bar_count_df["Size"], y=bar_count_df["Asks"], marker_color="#ff4d4d")
])
fig_bar_count.update_layout(
    barmode='group',
    title_text="Order Count by Size",
    xaxis_title="Order Size",
    yaxis_title="Count",
    height=320,
    template="plotly_dark",
    margin=dict(t=50, b=20),
)

# --- DISPLAY ---
st.subheader("Order Book (Top 100)")

col1, col2 = st.columns(2)
with col1:
    st.write("Bids (Buy Orders)")
    st.dataframe(pd.DataFrame(bids_display, columns=["Price", "Amount"], index=range(1, len(bids_display)+1)))
with col2:
    st.write("Asks (Sell Orders)")
    st.dataframe(pd.DataFrame(asks_display, columns=["Price", "Amount"], index=range(1, len(asks_display)+1)))

st.plotly_chart(fig_pie, use_container_width=True)
st.plotly_chart(fig_bar_vol, use_container_width=True)
st.plotly_chart(fig_bar_count, use_container_width=True)





refresh_interval = 60

# Toggle for enabling auto-refresh
enable_refresh = st.checkbox("🔁 Enable Auto-Refresh (every 60s)", value=True)

# Store the time of last refresh
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if enable_refresh:
    now = time.time()
    elapsed = now - st.session_state.last_refresh
    remaining = int(refresh_interval - elapsed)

    with st.empty():
        for seconds in range(refresh_interval, 0, -1):
            st.info(f"🔄 Auto-refreshing in {seconds} seconds...")
            time.sleep(1)
    st.session_state.last_refresh = now
    st.rerun()
