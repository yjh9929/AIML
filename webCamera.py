import cv2

capture = cv2.VideoCapture(0)


# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

while capture.isOpened():
    success, image = capture.read()
    cv2.imshow("VideoFrame", image)
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()