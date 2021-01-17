'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore
from ModelInferenceBase import ModelInferenceBase

class ModelFacialLandmarkDetection(ModelInferenceBase):
    '''
    Class for the Facial landmark detection Model.
    '''
       
    def predict(self, image):
        processed_image = self.preprocess_input(image)
        input_dict={self.input_name:processed_image}

        # result = self.exec_net.infer(input_dict)[self.output_name]

        self.exec_net.start_async(0, input_dict)

        if self.exec_net.requests[0].wait(-1) == 0:
            result = self.exec_net.requests[0].outputs[self.output_name]


        left_eye_image, right_eye_image, coords = self.preprocess_output(result, image)

        return left_eye_image, right_eye_image, coords

    def preprocess_input(self, image):
        *_, height, width = self.input_shape
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image=image.reshape(1, *image.shape)

        return image


    def preprocess_output(self, outputs, image):
        x_left = int(outputs[0][0][0] * image.shape[1])
        y_left = int(outputs[0][1][0] * image.shape[0])
        x_right = int(outputs[0][2][0] * image.shape[1])
        y_right = int(outputs[0][3][0] * image.shape[0])

        offset = 20

        x_left_min = x_left - offset
        y_left_min = y_left - offset
        x_left_max = x_left + offset
        y_left_max = y_left + offset

        x_right_min = x_right - offset
        y_right_min = y_right - offset
        x_right_max = x_right + offset
        y_right_max = y_right + offset

        coords = []
        coords.append((x_left_min, y_left_min, x_left_max, y_left_max))
        coords.append((x_right_min, y_right_min, x_right_max, y_right_max))

        # crop eyes
        eye_left_image = image[y_left_min:y_left_max, x_left_min:x_left_max]
        eye_right_image = image[y_right_min:y_right_max, x_right_min:x_right_max]

        return eye_left_image, eye_right_image, coords