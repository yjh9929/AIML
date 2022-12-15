import random
import os
import cv2

a = random.randint(0, 4)
b = random.randint(0, 1)
c = random.randint(0, 4)

if a+c < 9:
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
    cv2.waitKey(0)
    cv2.imshow("VideoFrame", opr)
    cv2.waitKey(0)
    cv2.imshow("VideoFrame", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Start again")