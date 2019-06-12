'''Wrapper package for OpenCV python bindings.
    Version 4.1.0.25
    {Open Source}'''
import cv2

'''Numpy is the fundamental package for array computing with python.
    Version 1.16.4
    {Open Source}'''
import numpy as np

'''Creat a CascadeClassifier Object
    CascadeClassifier contains the features of the face so that code can determine where is the face'''
faceDetect = cv2.CascadeClassifier('/home/oswalgaurav/PycharmProjects/numpy/haarcascade_frontalface_default.xml')

'''creating a video capture object.
    Value is 0 because we are using primary camera that is webacm.
    If there is a need to use the secondary camera we can change the value from 0 to 1'''
video = cv2.VideoCapture(0)

#loop until python is able to read the video object
while True:

    '''>check is a bool data type(Just like true and false), return true if python is able to read the video capture object.
        >frame is a numpy array, it represents the first image that video captures '''
    check, frame = video.read()

    # search the coordinates of the face in the image
    faces = faceDetect.detectMultiScale(frame, 1.05, 5)

    # converting the image into a gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''Method to create face rectangle.
        consist of image object that is "frame", RGB value of the rectangle outine and width of the rectangle'''
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)

        #cv2.imshow() to display an image in a window
        cv2.imshow('Face', frame)

        '''generate a new frame after every 1 miliseconds since we know that a video is nothing but multiple 
        frames that are displayed quickly'''
        key = cv2.waitKey(1)

    #Press key `q` to quit the program.
    if key == ord('q'):
        break

#When everything is done, release the capture.
video.release()

#close everything
cv2.destroyAllWindows()