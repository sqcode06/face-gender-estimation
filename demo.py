import data_utils as utils
from imutils.video import FileVideoStream, VideoStream
from imutils.video import FPS
from mtcnn import MTCNN
import time
import cv2
import model
import tensorflow as tf
import os
import csv

if __name__ == '__main__':
    arguments = utils.get_arguments()
    image_path = None
    video_path = None
    session = tf.compat.v1.Session()

    # load model multitask learning
    multitask_model = model.Model(session=session, trainable=False, prediction=True)

    # load model detect faces
    detect_model = MTCNN()

    directory = "..\\validation"

    rowlist = []

    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        print(image_path)

        actual_age = filename.split("_")[0]
        actual_gender = filename.split("_")[1]

        img = cv2.imread(image_path)

        # detect faces
        result = detect_model.detect_faces(img)

        # cropped face
        cropped_face, boxes = utils.crop_face(img, result)

        # predict
        images = (cropped_face - 128.0) / 255.0
        predicted_result = multitask_model.predict(images)

        if(predicted_result == []):
            predicted_result = [-1, -1]
        output = [actual_age, actual_gender]
        output.append(predicted_result[0])
        output.append(predicted_result[1])

        rowlist.append(output)

    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rowlist)
    file.close()
