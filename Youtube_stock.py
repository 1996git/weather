import json
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build

# 認証情報の読み込み
with open("secret.json") as f:
    secret = json.load(f)

DEVELOPER_KEY = secret["KEY"]
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# YouTube APIのクライアントを構築
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

def video_search(youtube, q="自動化", max_results=50):
    response = youtube.search().list(
        q=q,
        part="id,snippet",
        order="viewCount",
        type="video",
        maxResults=max_results
    ).execute()

    items_id = []
    for item in response.get("items", []):
        items_id.append({
            "video_id": item["id"]["videoId"],
            "channel_id": item["snippet"]["channelId"],
            "title": item["snippet"]["title"]
        })

    return pd.DataFrame(items_id)

def get_results(df_video, threshold=5000):
    channel_ids = df_video["channel_id"].unique()
    subscriber_list = youtube.channels().list(
        id=",".join(channel_ids),
        part="statistics",
        fields="items(id,statistics(subscriberCount))"
    ).execute()

    subscribers = []
    for item in subscriber_list.get("items", []):
        subscriber_count = int(item["statistics"].get("subscriberCount", 0))
        subscribers.append({
            "channel_id": item["id"],
            "subscriber_count": subscriber_count
        })

    df_subscribers = pd.DataFrame(subscribers)
    df = pd.merge(left=df_video, right=df_subscribers, on="channel_id")
    df_extracted = df[df["subscriber_count"] < threshold]

    video_ids = df_extracted["video_id"].tolist()
    video_info = []

    if video_ids:
        videos_list = youtube.videos().list(
            id=",".join(video_ids),
            part="snippet,statistics",
            fields="items(id,snippet(title),statistics(viewCount))"
        ).execute()

        for item in videos_list.get("items", []):
            video_info.append({
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "view_count": item["statistics"]["viewCount"]
            })

    df_video_info = pd.DataFrame(video_info)
    results = pd.merge(left=df_extracted, right=df_video_info, on="video_id", how='inner')

    results.rename(columns={'title_y': 'title'}, inplace=True)

    if 'title' in results.columns:
        results = results.loc[:, ["video_id", "title", "view_count", "subscriber_count", "channel_id"]]

    return results

# Streamlit アプリケーションの構成
st.title("YouTube動画検索アプリ")
st.sidebar.write("## クエリと閾値の設定")
query = st.sidebar.text_input("検索クエリを入力してください", "自動化")
threshold = st.sidebar.slider("登録者の閾値", 100, 10000, 5000)

st.write("### 選択中のパラメータ")
st.markdown(f"""
- 検索クエリ： {query}
- 登録者数の閾値: {threshold}
""")

video_field = st.empty()
video_field.write("こちらに動画が表示されます。")

if st.button("検索"):
    df_video = video_search(youtube, q=query)
    st.write("動画リスト:", df_video)

    results = get_results(df_video, threshold)

    if results.empty:
        st.write("条件に合う動画は見つかりませんでした。")
    else:
        st.write("検索結果:")
        st.dataframe(results)

        # 動画表示用の入力ボックス
        video_id = st.text_input("動画IDを入力してください", "")
        
        # 動画IDが選択されている場合に動画を表示
        if video_id:
            st.write("入力された動画ID:", video_id)
            st.write("検索結果の動画ID一覧:", results["video_id"].tolist())
            
            if video_id in results["video_id"].values:
                url = f"https://youtu.be/{video_id}"
                st.video(url)
            else:
                st.warning("入力した動画IDは検索結果にありません。")
                st.write("現在の検索結果:", results[["video_id", "title"]])