import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import keras.applications.xception as kax

IMAGE_SIZE = 64

st.title('WGAN-GP와  Xception 모델 변형을 통한 아토피 중증도 분류 모델 개선');
st.header('아토피의 중증도를 분류해줍니다. ');

image_file = st.file_uploader('이미지 업로드', type=['png','jpg','jpeg']);

# 이미지를 업로드 했다면 이미지를 보여준다. 
if(image_file):
    pilImage = Image.open(image_file);
    numpyImage=np.array(pilImage)
    st.image(pilImage);

    #이미지 전처리를 해줘야한다.
    processedImage = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2RGB);
    processedImage = cv2.resize(processedImage, (IMAGE_SIZE, IMAGE_SIZE));
    processedImage = kax.preprocess_input(processedImage);






