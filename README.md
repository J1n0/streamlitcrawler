# 구글 플레이 게임 리뷰 감정 분석기

## 프로젝트 배경

구글 플레이 스토어의 게임 리뷰를 수집하고, 감정 분석 모델을 통해 리뷰를 긍정, 부정, 중립으로 자동 분류하는 웹 애플리케이션을 구축하였습니다.

###[배포 주소](wlemail0095.streamlit.app)

###검색한 게임 리뷰 수집
![Image](https://github.com/user-attachments/assets/4002cc3e-02a3-460f-9cc7-d15523ebba4f)

###감정 분석 결과 시각화
![Image](https://github.com/user-attachments/assets/6e54e97d-986c-4c9a-8388-e1acd70d9c1b)
![Image](https://github.com/user-attachments/assets/4aa3629b-ce46-4bad-8b8a-c538211c918a)

### 데이터 수집량

수집량: 310,897개

### 라벨링 과정

google play store api로 크롤링한 리뷰파일을 lmstudio에 한줄 씩 넣어 감정분석 후 라벨링
사용 도구: gemma-lmstudio
감정 라벨: 긍정(1), 중립(0), 부정(-1)
사전 가공된 CSV 기준으로 학습 진행

# 모델 종류 및 학습 과정

모델: Logistic Regression + TF-IDF Vectorizer
정규표현식 기반 한글 토큰 추출 → Tokenizer: Okt 미사용
저장:simple_vectorizer_model.pkl


```하이퍼 파라미터값
TfidfVectorizer(max_features=5000)
LogisticRegression(max_iter=300)
test_size = 0.2
random_state = 42```

# 모델 성능

![Image](https://github.com/user-attachments/assets/0d94f884-8058-4287-aca1-800168ad4c34)

# 향후 개선 사항




