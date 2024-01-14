import cv2
import face_recognition
import os

# importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
studentId = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # print(os.path.splitext(path)[0])
    studentId.append(os.path.splitext(path)[0])

# print(studentId)

def encodeGen(imList):
    enco = []
    for image in imList:
        # try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        enco.append(encode)
        # except IndexError:
        #     # Handle the case where no face is found in the image
        #     print("No face found in an image.")
        #     encodings.append(None)

    return enco


print('start')
encodings = encodeGen(imgList)
