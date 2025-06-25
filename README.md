# 구글 플레이 게임 리뷰 감정 분석기

# 프로젝트 배경

구글 플레이 스토어의 게임 리뷰를 수집하고, 감정 분석 모델을 통해 리뷰를 긍정, 부정, 중립으로 자동 분류하는 웹 애플리케이션을 구축하였습니다.

# 프로젝트 구조
주요 기능
검색한 게임 리뷰 수집

감정 분석 결과 시각화

분석된 리뷰 CSV 다운로드 제공

# 필수 패키지 requirements.txt
pip install streamlit, pandas, scikit-learn, matplotlib, joblib, google-play-scraper

#실행 방법
wlemail0095.streamlit.app

# 데이터 수집

사용 API: google-play-scraper
수집 대상: 게임 앱 리뷰 (한글 기준)
데이터 종류: 게임이름, 리뷰 본문
수집량: 310,897개

# 라벨링 과정
사용 도구: gemma-lmstudio
감정 라벨: 긍정(1), 중립(0), 부정(-1)
사전 가공된 CSV 기준으로 학습 진행

# 모델 종류 및 학습 과정
모델: Logistic Regression + TF-IDF Vectorizer
정규표현식 기반 한글 토큰 추출 → Tokenizer: Okt 미사용
저장:simple_vectorizer_model.pkl
하이퍼 파라미터값
TfidfVectorizer(max_features=5000)
LogisticRegression(max_iter=300)
test_size = 0.2
random_state = 42

# 모델 성능
![Image](https://github.com/user-attachments/assets/0d94f884-8058-4287-aca1-800168ad4c34)

# 향후 개선 사항




