import streamlit as st
import yfinance as yf

# タイトル
st.title("株価データビジュアライザー")

# 銘柄の選択
ticker = st.selectbox("銘柄を選択してください", ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'])

# データの取得
data = yf.download(ticker, period='1d', interval='1m')

# データの表示
if not data.empty:
    st.write(data)

    # 線グラフの描画
    st.line_chart(data['Close'])
else:
    st.error("データが取得できませんでした。")
