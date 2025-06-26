# 구글 플레이 게임 리뷰 감정 분석기

## 프로젝트 배경

구글 플레이 스토어의 게임 리뷰를 수집하고, 감정 분석 모델을 통해 리뷰를 긍정, 부정, 중립으로 자동 분류하는 웹 애플리케이션을 구축하였습니다.

[배포 주소](wlemail0095.streamlit.app)

# 환경 설정

필수 패키지 requirements.txt

```pip install streamlit, pandas, scikit-learn, matplotlib, joblib, google-play-scraper```

# 실행 화면

검색한 게임 리뷰 수집
![Image](https://github.com/user-attachments/assets/4002cc3e-02a3-460f-9cc7-d15523ebba4f)

감정 분석 결과 시각화
![Image](https://github.com/user-attachments/assets/6e54e97d-986c-4c9a-8388-e1acd70d9c1b)
![Image](https://github.com/user-attachments/assets/4aa3629b-ce46-4bad-8b8a-c538211c918a)

데이터 수집량

수집량: 310,897개

라벨링 과정

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
random_state = 42
```

## 모델 성능

![Image](https://github.com/user-attachments/assets/0d94f884-8058-4287-aca1-800168ad4c34)

# 느낀 점

## 자동화의 한계와 검수의 중요성

리뷰 데이터를 수집하고 전처리하는 과정에서 자동화 도구를 활용했지만, 감정 라벨링이 완벽하지 않아 결국 사람이 직접 검수해야 하는 상황이 많았다. 
자동화로 시간을 아끼긴 했지만, 잘못된 라벨이 많아 검수 시간이 오히려 더 오래 걸렸다.
이 경험을 통해 자동화된 라벨링 결과를 사전에 일정 기준으로 필터링하거나, 신뢰도 기반으로 우선순위를 두는 방식이 필요하다는 점을 느꼈다.

모델의 확신이 낮은 데이터만 따로 분류해서 검수하는 방식을 도입하거나, 라벨링 전 소량의 데이터를 직접 학습시킨 커스텀 모델을 쓰는 방법을 써봐야겠다.
