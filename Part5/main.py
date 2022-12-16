# tensorflow 오류 잡기용
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import os
import cv2
from keras.models import load_model
import numpy as np
import sys

a = random.randint(0, 4)
b = random.randint(0, 1)
c = random.randint(0, 4)

print(a, b, c)

if (b == 0):
    os.chdir('Part5/picture')

    files = []

    file_names = os.listdir()
    for filename in file_names:
        if os.path.splitext(filename)[1] == '.PNG':
            files.append(filename)

    img1 = cv2.imread(files[a], cv2.IMREAD_COLOR)
    opr = cv2.imread(files[b+9], cv2.IMREAD_COLOR)
    img2 = cv2.imread(files[c], cv2.IMREAD_COLOR)

    cv2.imshow("VideoFrame", img1)
    cv2.waitKey(2000)
    cv2.imshow("VideoFrame", opr)
    cv2.waitKey(2000)
    cv2.imshow("VideoFrame", img2)

    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    ## 이미지 전처리
    def preprocessing(frame):
        # 크기 조정
        size = (224, 224)
        frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        # 이미지 정규화
        frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

        # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
        frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))

        return frame_reshaped

    # Load the model
    model = load_model('../keras_model.h5', compile=False)
    # Load the labels
    class_names = open('../labels.txt', 'r').readlines()

    # 카메라 캡처 객체, 0=내장 카메라
    capture = cv2.VideoCapture(0)


    # 캡쳐 프레임 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while capture.isOpened():
        success, image = capture.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # 이미지 뒤집기
        frame_fliped = cv2.flip(image, 1)

        # 이미지 출력
        if frame_fliped is None:
            print('Image load failed')
            sys.exit()
        cv2.imshow("VideoFrame", frame_fliped)

        if cv2.waitKey(1) == ord('q'):
            break
        # 데이터 전처리
        preprocessed = preprocessing(frame_fliped)

        # run the inference
        prediction = model.predict(preprocessed)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        expected_name = class_names[a+c+1]

        if class_name == expected_name and confidence_score >=0.7:
            #print('Class:', class_name, end='')
            #print('Confidence score:', confidence_score)
            print('the answer is: ', a+c+2)
            print('You are correct!')
            break

    capture.release()
    # 화면에 나타난 윈도우 창을 종료
    cv2.destroyAllWindows()
elif (b == 1 and a >= c):
    os.chdir('Part5/picture')

    files = []

    file_names = os.listdir()
    for filename in file_names:
        if os.path.splitext(filename)[1] == '.PNG':
            files.append(filename)

    img1 = cv2.imread(files[a], cv2.IMREAD_COLOR)
    opr = cv2.imread(files[b+9], cv2.IMREAD_COLOR)
    img2 = cv2.imread(files[c], cv2.IMREAD_COLOR)

    cv2.imshow("VideoFrame", img1)
    cv2.waitKey(3000)
    cv2.imshow("VideoFrame", opr)
    cv2.waitKey(3000)
    cv2.imshow("VideoFrame", img2)

    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    ## 이미지 전처리
    def preprocessing(frame):
        # 크기 조정
        size = (224, 224)
        frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        # 이미지 정규화
        frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

        # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
        frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))

        return frame_reshaped

    # Load the model
    model = load_model('../keras_model.h5', compile=False)
    # Load the labels
    class_names = open('../labels.txt', 'r').readlines()

    # 카메라 캡처 객체, 0=내장 카메라
    capture = cv2.VideoCapture(0)


    # 캡쳐 프레임 사이즈 조절
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while capture.isOpened():
        success, image = capture.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # 이미지 뒤집기
        frame_fliped = cv2.flip(image, 1)

        # 이미지 출력
        if frame_fliped is None:
            print('Image load failed')
            sys.exit()
        cv2.imshow("VideoFrame", frame_fliped)

        if cv2.waitKey(1) == ord('q'):
            break
        # 데이터 전처리
        preprocessed = preprocessing(frame_fliped)

        # run the inference
        prediction = model.predict(preprocessed)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        expected_name = class_names[a-c+1]

        if class_name == expected_name and confidence_score >=0.7:
            #print('Class:', class_name, end='')
            #print('Confidence score:', confidence_score)
            print('the answer is: ', a-c)
            print('You are correct!')
            break

    capture.release()
    # 화면에 나타난 윈도우 창을 종료
    cv2.destroyAllWindows()
else:
    print("Start again")