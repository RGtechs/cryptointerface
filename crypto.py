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
st.metric(label="Price", value=f"{quote_currency} {latest['close']:,.2f}")

# Συγκριτική τιμή από άλλα ανταλλακτήρια
exchanges_to_compare = {
    "Coinbase": ccxt.coinbase(),
    "Kraken": ccxt.kraken(),
    "Bitfinex": ccxt.bitfinex(),
    "Kucoin": ccxt.kucoin()
}

comparison_data = []

for name, ex in exchanges_to_compare.items():
    try:
        ex.load_markets()
        if symbol in ex.symbols:
            ticker = ex.fetch_ticker(symbol)
            price = ticker['last']
            diff = latest['close'] - price
            comparison_data.append({
                "Exchange": name,
                "Price": price,
                "Διαφορά με Binance": diff
            })
        else:
            comparison_data.append({
                "Exchange": name,
                "Price": "N/A",
                "Διαφορά με Binance": "N/A"
            })
    except Exception as e:
        comparison_data.append({
            "Exchange": name,
            "Price": "Error",
            "Διαφορά με Binance": str(e)
        })

# Προβολή πίνακα
st.subheader("Σύγκριση Τιμών με άλλα Ανταλλακτήρια")
st.dataframe(pd.DataFrame(comparison_data))

vol = sum(df['volume'])
st.metric(label="Volume", value=vol)

order_book = exchange.fetch_order_book(symbol)
bids = order_book['bids'][:]
asks = order_book['asks'][:]

st.subheader("Order Book (top 100)")
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
