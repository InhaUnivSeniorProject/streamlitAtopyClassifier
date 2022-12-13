import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import keras.applications.xception as kax
import keras.models as kmodel

IMAGE_SIZE = 64
myModel = kmodel.load_model('atopy_classifier1.h5');

st.title('WGAN-GP와  Xception 모델 변형을 통한 아토피 중증도 분류 모델 개선');
st.header('아토피의 중증도를 분류해줍니다. ');

xceptionBlockDiagram = Image.open('image/ourXception.png');

def show_grid_images(images_batch, ncols=4, title=None):
    figure, axs = plt.subplots(figsize=(22, 4), nrows=1, ncols=ncols)
    for i in range(ncols):
        # image_batch는 float형이므로 int형으로 변경하여 이미지 시각화
        axs[i].imshow(np.array(images_batch[i], dtype='int32'))
        axs[i].axis('off')
        axs[i].set_title(title[i]) 

def openImageAndConverToNumpy(ImagePath):
    pilImage = Image.open(ImagePath);                                
    numpyImage = np.array(pilImage);
    return numpyImage;

#이미지 불러오기
st.text("병변 별 gan으로 생성한 이미지입니다.");
myScratch11 = openImageAndConverToNumpy("image/gan/myScratch1_2355.jpg");
myScratch12 = openImageAndConverToNumpy("image/gan/myScratch1_2388.jpg");
myScratch13 = openImageAndConverToNumpy("image/gan/myScratch1_2394.jpg");
myScratch21 = openImageAndConverToNumpy("image/gan/myScratch2_2356.jpg");
myScratch22 = openImageAndConverToNumpy("image/gan/myScratch2_2373.jpg");
myScratch23 = openImageAndConverToNumpy("image/gan/myScratch2_2392.jpg");
myScratch31 = openImageAndConverToNumpy('image/gan/myScratch3__2161.jpg');
myScratch32 = openImageAndConverToNumpy('image/gan/myScratch3__2217.jpg'); 
myScratch33 = openImageAndConverToNumpy('image/gan/myScratch3__2265.jpg');
myLi11 = openImageAndConverToNumpy('image/gan/myLi1_2383.jpg');
myLi12 = openImageAndConverToNumpy('image/gan/myLi1_2392.jpg');
myLi13 = openImageAndConverToNumpy('image/gan/myLi1_2397.jpg');
myLi21 = openImageAndConverToNumpy('image/gan/myLi2_2387.jpg');
myLi22 = openImageAndConverToNumpy('image/gan/myLi2_2388.jpg');
myLi23 = openImageAndConverToNumpy('image/gan/myLi2_2392.jpg');
myLi31 = openImageAndConverToNumpy('image/gan/myLi3_1890.jpg');
myLi32 = openImageAndConverToNumpy('image/gan/myLi3_1893.jpg');
myLi33 = openImageAndConverToNumpy('image/gan/myLi3_1908.jpg');
myRed31 = openImageAndConverToNumpy('image/gan/myReds_2280.jpg');
myRed32 = openImageAndConverToNumpy('image/gan/myReds_2332.jpg');
myRed33 = openImageAndConverToNumpy('image/gan/myReds_2382.jpg');
myRed11 = openImageAndConverToNumpy('image/gan/myReds_2360.jpg');
myRed12 = openImageAndConverToNumpy('image/gan/myReds_2342.jpg');
myRed13 = openImageAndConverToNumpy('image/gan/myReds_2317.jpg');
myRed21 = openImageAndConverToNumpy('image/gan/myReds_2326.jpg');
myRed22 = openImageAndConverToNumpy('image/gan/myReds_2353.jpg');
myRed23 = openImageAndConverToNumpy('image/gan/myReds_2298.jpg');

col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
col1.write("홍진1");
col1.image(myRed11)
col1.image(myRed12)
col1.image(myRed13)
col2.write("홍진2");
col2.image(myRed21);
col2.image(myRed22);
col2.image(myRed23);
col3.write("홍진3")
col3.image(myRed31);
col3.image(myRed32);
col3.image(myRed33);
col4.write("상처1");
col4.image(myScratch11)
col4.image(myScratch12)
col4.image(myScratch13)
col5.write("상처2")
col5.image(myScratch21)
col5.image(myScratch22)
col5.image(myScratch23)
col6.write("상처3")
col6.image(myScratch31);
col6.image(myScratch32);
col6.image(myScratch33);
col7.write("태선화1");
col7.image(myLi11);
col7.image(myLi12);
col7.image(myLi13);
col8.write("태선화2");
col8.image(myLi21);
col8.image(myLi22);
col8.image(myLi23);
col9.write("태선화3");
col9.image(myLi31);
col9.image(myLi32);
col9.image(myLi33);

st.text("**변형된 Xception 모델의 블록도 입니다.**")
st.image(xceptionBlockDiagram);

# 이미지를 업로드 했다면 이미지를 보여준다. 
image_file = st.file_uploader('이미지 업로드', type=['png','jpg','jpeg']);


if(myModel):
    st.text(myModel)
if(image_file):
    pilImage = Image.open(image_file);
    numpyImage=np.array(pilImage)
    st.image(pilImage);

    #이미지 전처리를 해줘야한다.
    processedImage = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2RGB);
    processedImage = cv2.resize(processedImage, (IMAGE_SIZE, IMAGE_SIZE));
    processedImage = kax.preprocess_input(processedImage);
    processedImage = np.reshape(processedImage, (1,64,64,3))

    
    summary = myModel.summary(); 
    # 태선화1 2 3 홍진1 2 3 긁은 상처 1 2 3 순서 
    print(myModel.predict(processedImage))
    result = myModel.predict(processedImage)[0].argmax()

    def switch(key):
        myLabel = {"0" : "태선화1", "1" : "태선화2","2" : "태선화3","3":"홍진1","4" : "홍진2", "5" : "홍진3", "6" : "긁은상처1", "7" : "긁은상처2","8": "긁은상처3"}.get(key)
        st.header("병변 및 중증도");
        st.header(myLabel)

    switch(str(result));

    






