import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('jr.jpg')
#img = cv2.imread('2people.jpg')
#to capture vedio from webcam
webcam = cv2.VideoCapture(0)

while True:
    #read the current frame
    successful_frame_read,frame = webcam.read()

    #converting to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles around faces
    for (x,y,w,h) in face_coordinates:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),10)

#print(face_coordinates)

    cv2.imshow('Sudam Ghost Face Detector', frame)
    key = cv2.waitKey(1)

    #stop if Q or q is pressed
    if key==81 or key==113:
        break

print("Code Completed")