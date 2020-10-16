import cv2
import numpy as np
import traceback
from tensorflow.keras import utils
from dataloader.augmentor import augment
from dataloader import preprocess as preprocess_utils

class DataGenerator(utils.Sequence):

    'Generates data for Keras'
    def __init__(self, all_data, class_count, config, img_length_calc_function, mode='train', shuffle = True):
        'Initialization'
        self.all_data = all_data
        self.class_count = class_count
        self.config = config
        self.img_length_calc_function = img_length_calc_function
        self.shuffle = shuffle
        self.mode = mode
        self.batch_size = 1
        self.on_epoch_end()
        self.indexes = np.arange(len(self.all_data))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.all_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        temp_data = [self.all_data[k] for k in indexes]
        assert len(temp_data) == 0 # current implimentation is only for batch = 1
        # Generate data
        X, y, meta = self.__data_generation(temp_data[0])

        return X, y, meta

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, single_data):

        if self.mode == 'train':
            img_data_aug, x_img = augment(single_data, self.config, augment=True)
        else:
            img_data_aug, x_img = augment(single_data, self.config, augment=False)

        (width, height) = (img_data_aug['width'], img_data_aug['height'])
        (rows, cols, _) = x_img.shape

        assert cols == width, "annotation width and image width doesn't match %s" % img_data_aug['filepath'].name
        assert rows == height, "annotation height and image height doesn't match %s" % img_data_aug['filepath'].name

        # get image dimensions for resizing
        (resized_width, resized_height) = preprocess_utils.get_new_img_size(width, height, self.config.im_size)

        # resize the image so that smalles side is length = 600px
        x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

        try:
            y_rpn_cls, y_rpn_regr = preprocess_utils.calc_rpn(self.config, img_data_aug, width, height, resized_width,
                                                              resized_height, self.img_length_calc_function)
        except Exception:
            print("Error in calculating RPN: ", str(traceback.format_exc()))
            raise Exception("issue in dataset")
        # Zero-center by mean pixel, and preprocess image

        x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
        x_img = x_img.astype(np.float32)
        x_img[:, :, 0] -= self.config.img_channel_mean[0]
        x_img[:, :, 1] -= self.config.img_channel_mean[1]
        x_img[:, :, 2] -= self.config.img_channel_mean[2]
        x_img /= self.config.img_scaling_factor

        x_img = np.expand_dims(x_img, axis=0)

        y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= self.config.std_scaling

        return np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug