import cv2

img = cv2.imread("/home/oswalgaurav/Downloads/messi.jpg",1)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

resized = cv2.resize(gray_img, (int(img.shape[1]/2),int(img.shape[0]/2)))

cv2.imshow("Gray",resized)

cv2.waitKey(0)

cv2.destroyAllWindows()