# pip install opencv-python
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from os import mkdir
from os import chdir
from os.path import exists


def train_model(model_train, username):
    print('Started model training!')

    data_path = '/home/oswalgaurav/Downloads/messi/' + username + '/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)
    # linear binary phase histogram face recognizer
    # model = cv2.face.LBPHFaceRecognizer_create()

    model_train.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Complete!!!!!")

    return model_train


def add_face(username):
    face_classifier_obj = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # cv2.VideoCapture( camera ID ) - starts camera(webcam)
    # type(capture) = VideoCapture obj
    video_capture_obj = cv2.VideoCapture(0)

    count = 0

    while True:

        # video_capture_obj.read() - reads a video_frame
        # type(camera_frame) - numpy_array
        retuurn, camera_frame = video_capture_obj.read()

        # cv2.cvtColor(image,format_to_convert) - converts an image to a specified format
        # type(gray_image) - numpy array
        gray_image = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)

        # face_classifier_obj.detectMultiScale(image,scaling_factor_neighbours) - returns face coordinates of the image else none
        # type(image_cropped_coordinates_xywh) - tuple
        image_cropped_coordinates_xywh = face_classifier_obj.detectMultiScale(gray_image, 1.3, 5)

        if (len(image_cropped_coordinates_xywh) != 0):
            for [x, y, w, h] in image_cropped_coordinates_xywh:
                pass

            count += 1

            # cropping the image
            # typr(cropped_gray_image) - numpy array
            cropped_gray_image = gray_image[y:y + h, x:x + w]

            # cv2.resize(image, tuple(x_pixels,y_pixels) - resizing the image
            # typr(resized_cropped_gray_image) - numpy array
            resized_cropped_gray_image = cv2.resize(cropped_gray_image, (500, 500))

            chdir('face_data')
            if not exists(username):
                mkdir(username)
            chdir('..')
            # cv2.imwrite( 'path to save image', image) - saves the image
            cv2.imwrite('/home/oswalgaurav/Downloads/messi/' + username + '/user_' + str(
                count) + '.jpg', resized_cropped_gray_image)

            # cv2.putText(image, string_text, image_bottom_right_corner, font, font_scale, tuple(color), thickness)
            # puts text on the image
            cv2.putText(resized_cropped_gray_image, 'image no. - ' + str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0), 2)

            # cv2.imshow( 'window name' ,image ) - shows image in an window
            cv2.imshow('Cropped Image Show', resized_cropped_gray_image)

        else:
            print("Face not detected")

        # cv2.waitKey( delay milli_seconds) - waits for a key press
        # returns the ascii value of the key
        if (cv2.waitKey(1) == 13 or count == 300):
            break

    # releases camera
    video_capture_obj.release()

    # deletes all cv2 windows
    cv2.destroyAllWindows()

    print('Face data collection completed :) ')


def detect_face(username):
    # ?
    face_classifier_obj = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # linear binary phase histogram face recognizer
    model = cv2.face.LBPHFaceRecognizer_create()

    chdir('face_data')
    chdir(username)
    if not exists(username + ' model'):
        add_face(username)
        mkdir(username + ' model')
        chdir(username + ' model')
        model = train_model(model, username)
        model.save(username + ' model')
    else:
        chdir(username + ' model')
        model.read(username + ' model')
    chdir('..')
    chdir('..')
    chdir('..')

    video_capture_1 = cv2.VideoCapture(0)

    wrong_face = 0
    right_face = 0
    temp = 0

    red_color = 1

    while True:
        ret_1, camera_frame_1 = video_capture_1.read()

        gray_image_1 = cv2.cvtColor(camera_frame_1, cv2.COLOR_BGR2GRAY)

        image_cropped_coordinates_xywh_1 = face_classifier_obj.detectMultiScale(gray_image_1, 1.3, 5)

        if (len(image_cropped_coordinates_xywh_1) != 0):

            for x_1, y_1, w_1, h_1 in image_cropped_coordinates_xywh_1:

                # cv2.rectangle(camera_frame_1, (x_1,y_1), (x_1+w_1,y_1+h_1), (0,((not red_color)*255),(red_color*255)), 2)

                cropped_gray_image_1 = gray_image_1[y_1:y_1 + h_1, x_1:x_1 + w_1]

                resized_cropped_gray_image_1 = cv2.resize(cropped_gray_image_1, (500, 500))

                result = model.predict(resized_cropped_gray_image_1)

                # print(len(image_cropped_coordinates_xywh_1))

                if (result[1] < 500):
                    confidence = int(100 * (1 - (result[1] / 300)))
                    cv2.putText(camera_frame_1, str(result[1]) + ' , ' + str(confidence) + '% confidence it is user',
                                (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    if (confidence > 90):
                        cv2.putText(camera_frame_1, 'YOU ARE ' + username, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 255, 0), 2)
                        right_face += 1
                        red_color = 0
                        print('.............................')
                    else:
                        cv2.putText(camera_frame_1, 'YOU ARE NOT ' + username, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 0, 255), 2)
                        wrong_face += 1
                        temp += 1
                        red_color = 1
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                cv2.rectangle(camera_frame_1, (x_1, y_1), (x_1 + w_1, y_1 + h_1),
                              (0, ((not red_color) * 255), (red_color * 255)), 2)

        else:
            cv2.putText(camera_frame_1, 'NO USER Detected', (450, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Face Recognizer :)', camera_frame_1)

        if (cv2.waitKey(1) == 13):
            break

        if (right_face >= 2):
            wrong_face = 0

        if (wrong_face >= 100 or temp >= 400):
            video_capture_1.release()
            cv2.destroyAllWindows()
            return 'errer'
        if (right_face >= 30):
            video_capture_1.release()
            cv2.destroyAllWindows()
            return 'face_matched'
