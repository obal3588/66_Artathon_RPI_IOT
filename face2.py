import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
key = cv2. waitKey(1)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num=0
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(laplacian, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            cv2.circle(laplacian, (x, y), 6, (255, 0, 0), -10)
            
        if key == ord('s'):
            num=num+1
            newIma=cv2.resize(frame,(512,512))
            cv2.imwrite(filename=('saved_img(%d).jpg' % num), img=frame)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    cv2.imshow("Frame", frame)
    cv2.imshow("jj", laplacian)

    key = cv2.waitKey(1)
    if key == 27:
        break