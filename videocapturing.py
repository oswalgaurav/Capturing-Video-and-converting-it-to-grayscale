import cv2, time

#creating a video capture object
video = cv2.VideoCapture(0)

#tell us the number of frames
a=1

#loop until python is able to read the video object
while True:
    a = a + 1
    check,frame = video.read()
    print(frame)
    #converting the image into a gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    cv2.imshow("Capture",gray)
    key = cv2.waitKey(1) #generate a new frame after every 1 miliseconds
    if key == ord('q'):
        break

print(a)

video.release()

cv2.destroyAllWindows()