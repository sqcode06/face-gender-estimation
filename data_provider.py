import numpy as np

import data_utils
import config as cf
import utils


class Datasets(object):
    def __init__(self, trainable, test_data_type='public_test'):
        self.all_data = []
        self.trainable = trainable
        self.smile_train, self.smile_test = data_utils.getSmileImage()
        self.age_train, self.age_test = data_utils.getAgeImage()
        self.gender_train, self.gender_test = data_utils.getGenderImage()

        if not trainable:
            self.test_data_type = test_data_type

        self.convert_data_format()

    def gen(self):
        np.random.shuffle(self.all_data)
        batch_images = []
        batch_labels = []
        batch_indexes = []

        for i in range(len(self.all_data)):
            image, label, index = self.all_data[i]
            batch_images.append(image)
            batch_labels.append(label)
            batch_indexes.append(index)

            if len(batch_images) == cf.BATCH_SIZE:
                yield batch_images, batch_labels, batch_indexes
                batch_images = []
                batch_labels = []
                batch_indexes = []

        if len(batch_images) > 0:
            yield batch_images, batch_labels, batch_indexes

    def convert_data_format(self):
        if self.trainable:
            # Smile dataset
            for i in range(len(self.smile_train) * 10):
                image = (self.smile_train[i % 3000][0] - 128.0) / 255.0
                label = utils.get_one_hot_vector(7, int(self.smile_train[i % 3000][1]))
                index = 1.0
                self.all_data.append((image, label, index))
        
            # Age datasets
            for i in range(len(self.age_train)):
                image = (self.age_train[i][0] - 128.0) / 255.0
                label = utils.get_one_hot_vector(7, int(self.age_train[i][1]))
                index = 3.0
                self.all_data.append((image, label, index))
           
            # Gender datasets
            for i in range(len(self.gender_train)):
                image = (self.gender_train[i][0] - 128.0) / 255.0
                label = utils.get_one_hot_vector(7, int(self.gender_train[i][1]))
                index = 4.0
                self.all_data.append((image, label, index))

        else:
            # Smile datasets
            for i in range(len(self.smile_test)):
                image = (self.smile_test[i][0] - 128.0) / 255.0
                label = utils.get_one_hot_vector(7, int(self.smile_test[i][1]))
                index = 1.0
                self.all_data.append((image, label, index))
       
            # Age datasets
            for i in range(len(self.age_test)):
                image = (self.age_test[i][0] - 128.0) / 255.0
                label = utils.get_one_hot_vector(7, int(self.age_test[i][1]))
                index = 3.0
                self.all_data.append((image, label, index))
        
            # Gender datasets
            for i in range(len(self.gender_test)):
                image = (self.gender_test[i][0] - 128.0) / 255.0
                label = utils.get_one_hot_vector(7, int(self.gender_test[i][1]))
                index = 4.0
                self.all_data.append((image, label, index))
