# game_review_sentiment_app.py
import streamlit as st
import pandas as pd
import joblib
from google_play_scraper import app, reviews, search
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# 형태소 분석기 정의
def tokenize(text):
    okt = Okt()
    return okt.morphs(text, stem=True)

# 모델 불러오기
clf, vectorizer = joblib.load("okt_vectorizer_model.pkl")

# Streamlit 페이지 설정
st.set_page_config(page_title="게임 리뷰 감정 분석기", layout="wide")
st.title("🎮 구글 플레이 게임 리뷰 감정 분석기")

# 검색어로 앱 리스트 동적 검색
if 'selected_app_id' not in st.session_state:
    st.session_state.selected_app_id = None
if 'selected_game' not in st.session_state:
    st.session_state.selected_game = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'checkbox_state' not in st.session_state:
    st.session_state.checkbox_state = {
        'pie': False,
        'bar': False
    }

# 검색 입력
game_query = st.text_input("게임 이름을 입력하세요")

app_options = []
app_ids = {}

if game_query:
    results = search(game_query, lang='ko', country='kr')
    for result in results[:30]:
        name = result['title']
        app_id = result['appId']
        app_options.append(name)
        app_ids[name] = app_id

selected_game = st.selectbox("게임을 선택하세요", app_options)
selected_app_id = app_ids.get(selected_game)
st.session_state.selected_game = selected_game
st.session_state.selected_app_id = selected_app_id

# 감정 분석 함수
def predict_sentiment(texts):
    if len(texts) == 0:
        return []
    X = vectorizer.transform(texts)
    return clf.predict(X)

# 리뷰 수집 함수
def crawl_reviews(app_id, max_count=200):
    try:
        result, _ = reviews(
            app_id,
            lang='ko',
            country='kr',
            count=max_count
        )
        return [r['content'] for r in result if r['content'].strip() != '']
    except Exception as e:
        st.error(f"리뷰 수집 실패: {e}")
        return []

if selected_app_id and st.button("리뷰 수집 및 감정 분석"):
    with st.spinner("리뷰 수집 중..."):
        reviews_list = crawl_reviews(selected_app_id)

    if not reviews_list:
        st.warning("리뷰를 수집하지 못했습니다.")
    else:
        st.success(f"{len(reviews_list)}개의 리뷰 수집 완료")

        df = pd.DataFrame({"리뷰": reviews_list})
        df['감정'] = predict_sentiment(df['리뷰'])
        df['감정'] = df['감정'].map({1: '긍정', 0: '중립', -1: '부정'})
        st.session_state.df = df
        st.dataframe(df)

# 감정 분석 시각화 (세션에서 유지)
df = st.session_state.df
if df is not None:
    pie_checked = st.checkbox("감정 분석 비율 보기", value=st.session_state.checkbox_state['pie'])
    st.session_state.checkbox_state['pie'] = pie_checked

    if pie_checked:
        fig, ax = plt.subplots()
        df['감정'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, labels=df['감정'].value_counts().index, textprops={'fontsize': 14})
        ax.set_ylabel('')
        ax.set_title(f"'{st.session_state.selected_game}' 감정 비율")
        st.pyplot(fig)

    bar_checked = st.checkbox("감정 분석 막대 그래프 보기", value=st.session_state.checkbox_state['bar'])
    st.session_state.checkbox_state['bar'] = bar_checked

    if bar_checked:
        fig2, ax2 = plt.subplots()
        df['감정'].value_counts().plot.bar(color=['red', 'gray', 'green'], ax=ax2)
        ax2.set_ylabel('리뷰 수')
        ax2.set_title(f"'{st.session_state.selected_game}' 감정 분석 결과")
        st.pyplot(fig2)

    # 감정별 리뷰 필터링
    st.subheader("감정별 리뷰 필터")
    sentiment_filter = st.radio("감정 선택", options=['전체', '긍정', '중립', '부정'], horizontal=True)
    if sentiment_filter != '전체':
        filtered_df = df[df['감정'] == sentiment_filter]
    else:
        filtered_df = df

    st.dataframe(filtered_df)

    # 다운로드 버튼
    csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 필터링된 리뷰 CSV 다운로드",
        data=csv,
        file_name=f"{st.session_state.selected_game}_{sentiment_filter}_리뷰.csv",
        mime='text/csv'
    )
