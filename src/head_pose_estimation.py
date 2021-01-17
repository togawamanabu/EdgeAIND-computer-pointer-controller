'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore
from ModelInferenceBase import ModelInferenceBase

import logging as log

class ModelHeadPoseEstimation(ModelInferenceBase):
    '''
    Class for the Head pose estimation Model.
    '''

    def predict(self, image):
        processed_image = self.preprocess_input(image)
        input_dict={self.input_name:processed_image}

        # result = self.exec_net.infer(input_dict)

        self.exec_net.start_async(0, input_dict)

        if self.exec_net.requests[0].wait(-1) == 0:
            result = self.exec_net.requests[0].outputs


        yaw, pitch, roll = self.preprocess_output(result, image)

        return yaw, pitch, roll

    def preprocess_input(self, image):
        *_, height, width = self.input_shape
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image=image.reshape(1, *image.shape)

        return image

    def preprocess_output(self, outputs, image):
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]

        return yaw, pitch, roll