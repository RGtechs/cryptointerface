import streamlit as st
import ccxt
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Real-time BTC/USDT Dashboard")

exchange = ccxt.binance()

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
    "999 Days": ("1d", timedelta(days=999)),
}

selected = st.selectbox("Select Time Range", list(timeframes.keys()), index=0)
tf, delta = timeframes[selected]

now = exchange.milliseconds()
if delta is not None:
    since = now - int(delta.total_seconds() * 1000)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since)
else:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=5000)

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

y_min = 0.99999 * df['low'].min()
y_max = 1.00001 * df['high'].max()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='BTC Close Price'))

fig.update_layout(
    title=f"BTC/USDT Price ({selected})",
    yaxis=dict(title="Price ("+ quote_currency + ")", range=[y_min, y_max], fixedrange=True),
    xaxis=dict(title="Time", fixedrange=True),
    template='plotly_dark',
    height=400,
    dragmode=False,
    margin=dict(l=0, r=0, t=40, b=0),
    modebar_remove=["zoom", "zoomIn", "zoomOut", "autoScale", "pan", "resetScale"]
)

st.plotly_chart(fig, use_container_width=True)

latest = df.iloc[-1]
st.metric(label="Price (USD)", value=f"${latest['close']:,.2f}")
vol = sum(df['volume'])
st.metric(label="Volume", value=vol)

order_book = exchange.fetch_order_book(symbol)
bids = order_book['bids'][:]
asks = order_book['asks'][:]

st.subheader("Order Book (top 500)")
col1, col2 = st.columns(2)

with col1:
    st.write("**Bids (Buy Orders)**")
    st.dataframe(pd.DataFrame(bids, columns=["Price", "Amount"], index=range(1, len(bids)+1)))

with col2:
    st.write("**Asks (Sell Orders)**")
    st.dataframe(pd.DataFrame(asks, columns=["Price", "Amount"], index=range(1, len(asks)+1)))

import plotly.express as px

total_bid_volume = sum([amount for _, amount in bids])
total_ask_volume = sum([amount for _, amount in asks])

pie_df = pd.DataFrame({
    "Side": ["🟢 Bids (Buy Volume)", "🔴 Asks (Sell Volume)"],
    "Volume": [total_bid_volume, total_ask_volume]
})

fig_pie = px.pie(
    pie_df,
    names="Side",
    values="Volume",
    hole=0.5,
    color="Side",
    color_discrete_map={
        "🟢 Bids (Buy Volume)": "#00cc96",
        "🔴 Asks (Sell Volume)": "#ff4d4d"
    }
)

fig_pie.update_traces(
    textposition="outside",
    textinfo="label+percent",
    marker=dict(line=dict(color='#1e1e1e', width=2)),
    pull=[0.02, 0.02]
)

fig_pie.update_layout(
    title_text="📊 Order Book Volume Split",
    title_font_size=18,
    title_x=0.0,
    showlegend=False,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font_color="white",
    margin=dict(t=60, b=20, l=0, r=0),
    height=320
)

st.plotly_chart(fig_pie, use_container_width=True)
