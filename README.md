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
3. gan으로ㅓ 생성한 이미지 보여주기 
![image](https://user-images.githubusercontent.com/71626430/207265888-71a04805-a5cd-474b-882e-821e12ad32bc.png)
4. 파일 업로드하면 아토피 중증도 분류해주기

## 배포된 주소 
https://inhaatopyclassifier.streamlit.app/
