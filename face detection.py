import cv2, time

first_frame = None

#Create a videocapture object to record video using webcam
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    #convert the frame color to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #convert the gray scale frame to GaussianBlur to increase the accuracy of the detected frame
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    #this is used to store he first image/frame of the video
    if first_frame is None:
        first_frame = gray
        continue

    #Calculate the difference between the first frame and the other frames
    delta_frame = cv2.absdiff(first_frame, gray )

    #Provides a threshold value, such that it will convert the difference value with less than 30to black.
    #If the difference is greater than 30 it will convert those pixels to white
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations= 0)

    #Define the counter area. Basically, add the borders
    (_,cnts,_) = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Remove Noises nd Shadows. Basically, it will keep only that part white, which has greater white area than 1000 pixels
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('frame', frame)
    cv2.imshow('Capturing', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh',thresh_delta)

    #frame will change in 1 milisecond
    key = cv2.waitKey(1)

    #this will break the loop once the user press 'q'
    if key == ord('q'):
        break

video.release()

#this will close all the windows
cv2.destroyAllWindows()

