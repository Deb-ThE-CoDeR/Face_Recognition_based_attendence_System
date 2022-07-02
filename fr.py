import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images'
images = []
personName = []
mylist = os.listdir(path)
print(mylist)
for cu_img in mylist:
    current_img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = faceEncodings(images)
print("----------------------------")
print("All encodings are completed")
print("----------------------------")


def attendence(name):
    with open('att.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split('/n')
            nameList.append(entry[0])
        if name not in nameList:
            time_now=datetime.now()
            tstr = time_now.strftime('%H:%M:%S')
            dstr = time_now.strftime('%d:%m:%y')
            f.writelines(f'\n{name},{tstr},{dstr}')

#attendence('Anu')

cap=cv2.VideoCapture(0)     #0 for laptop camera and for external camera use 1

while True:
     ret, frame = cap.read()   #read the camera frame
     faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
     faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

     faceCurrentFrame = face_recognition.face_locations(faces)
     encodesCurrentFrame = face_recognition.face_encodings(faces,faceCurrentFrame)

     for encodeFace,faceloc in zip(encodesCurrentFrame, faceCurrentFrame):
         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
         match_index = np.argmin(faceDis)

         if matches[match_index]:
             name = personName[match_index].upper()

             y1, x2, y2, x1 = faceloc
             y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
             cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
             attendence(name)

     cv2.imshow("camera", frame)
     if cv2.waitKey(1) == 13:
         break

cap.release()
cv2.destroyAllWindows()
