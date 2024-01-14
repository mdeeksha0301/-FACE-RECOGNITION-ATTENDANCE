import os
import pickle
import face_recognition
import numpy as np
import cvzone
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for modePath in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, modePath)))

# load the encoding file
print('Loading encode file..')
file = open('EncodeGen.p', 'rb')
encodingsWithIds = pickle.load(file)
file.close()
encodings, studentId = encodingsWithIds
# print(studentId)
print('Encode file loaded..')

while True:
    success, img = cap.read()

    imageS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imageS = cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imageS)
    encodeCurrFrame = face_recognition.face_encodings(imageS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]

    for encodeFace, faceloca in zip(encodeCurrFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodings, encodeFace)
        faceDis = face_recognition.face_distance(encodings, encodeFace)
        # print("matches", matches)
        # print("face distance", faceDis)

        matchIndex = np.argmin(faceDis)
        # print(matchIndex)

        if matches[matchIndex]:
            # print('Match found!')
            # print(studentId[matchIndex])
            y1, x2, y2, x1 = faceloca
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)


    # cv2.imshow('img', img)
    cv2.imshow('imgBackground', imgBackground)
    cv2.waitKey(1)
