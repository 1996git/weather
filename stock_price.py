import pandas as pd
import yfinance as yf
import altair as alt
import streamlit as st

st.title("米国株価可視化アプリ")

st.sidebar.write("""
# GAFA株価
こちらは株価可視化ツールです。以下のオプションから表示選択を指定してください。
""")

# 期間の選択肢を指定
period_options = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
selected_period = st.sidebar.selectbox("期間を選択してください", period_options)

st.write(f"""
### 過去 **{selected_period}** のGAFA株価
""")

@st.cache_data
def get_data(period, tickers):
    df = pd.DataFrame()
    for company, ticker in tickers.items():
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period=period)

            if hist.empty:
                st.warning(f"{company} のデータは取得できませんでした。")
                continue

            hist.index = hist.index.strftime("%d %B %Y")
            hist = hist[["Close"]]
            hist.columns = [company]
            df = pd.concat([df, hist.T])
        except Exception as e:
            st.error(f"{company} のデータ取得中にエラーが発生しました: {e}")
    return df

st.sidebar.write("""
## 株価の範囲指定
""")
ymin, ymax = st.sidebar.slider(
    "範囲を指定してください",
    0.0, 1000.0, (0.0, 1000.0)
)

tickers = {
    "Apple": "AAPL",
    "Facebook": "META",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Amazon": "AMZN"
}

df = get_data(selected_period, tickers)

available_companies = list(df.index)

companies = st.multiselect(
    "会社名を選択してください",
    available_companies,
    available_companies[:4]
)

if not companies:
    st.error("少なくとも一社は選んでください")
else:
    data = df.loc[companies]
    st.write("### 株価（USD）", data.sort_index())
    
    # データの確認
    st.write("データフレームの内容:", data)

    # データを整形
    data = data.T.reset_index()
    data.columns = ['Date'] + list(data.columns[1:])

    # データをロング形式に変換
    data = pd.melt(data, id_vars=["Date"]).rename(
        columns={"value": "株価(USD)"}
    )

    # データの確認
    st.write("ロング形式データ:", data)

    # グラフ作成
    chart = (
        alt.Chart(data)
        .mark_line(opacity=0.8, clip=True)
        .encode(
            x="Date:N",
            y=alt.Y("株価(USD):Q", stack=None, scale=alt.Scale(domain=[ymin, ymax])),
            color="variable:N"
        )
    )

    st.altair_chart(chart, use_container_width=True)