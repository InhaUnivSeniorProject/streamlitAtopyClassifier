# streamlit을 이용한 간단한 모델 배포

## 파이썬 설치 후

### 패키지 설치
```
pip install streamlit
pip install tensorflow
pip install opencv-python
pip install matplotlib
```

### 브라우저에 띄우기
```
streamlit run main.py
```

### 위의 명령어가 듣지 않을 시 
```
python -m streamlit run main.py
```

### 현재 구현된 기능 
1. 파일 업로드 후 보여주기 
2. 이미지 전처리
![image](https://user-images.githubusercontent.com/71626430/207253728-95d05158-1a2a-4ac4-b954-adf93249d31b.png)


### 구현해야할 기능
1. 모델 업로드 
2. GAN으로 생성한 이미지 matplotlib으로 보여주기
3. 모델에 전처리된 이미지를 통과시킨 후 결과 출력 
