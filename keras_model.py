# tensorflow 오류 잡기용
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 모듈 불러오기
import cv2
import tensorflow.keras
import numpy as np

capture = cv2.VideoCapture(0)

## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    
    return frame_reshaped

## 학습된 모델 불러오기
model_filename = 'keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename)

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while capture.isOpened():
    success, image = capture.read()
    cv2.imshow("VideoFrame", image)
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()

