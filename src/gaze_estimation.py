'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore
from ModelInferenceBase import ModelInferenceBase

import logging as log 

class ModelGazeEstimation(ModelInferenceBase):
    '''
    Class for Gaze estimation Model.
    '''

    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        processed_lefteye_image = self.preprocess_input(left_eye_image)
        processed_righteye_image = self.preprocess_input(right_eye_image)

        input_dict={
                    'left_eye_image': processed_lefteye_image,
                    'right_eye_image': processed_righteye_image,
                    'head_pose_angles': head_pose_angle
                    }

        result = self.exec_net.infer(input_dict)[self.output_name]

        coords = self.preprocess_output(result)

        return coords

    def check_model(self):
        # log.debug(self.net.inputs)

        self.input_name=[i for i in self.net.inputs.keys()]

        # log.debug(self.input_name)
        self.input_shape=self.net.inputs[self.input_name[1]].shape

        # log.debug(self.input_shape)
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape


    def preprocess_input(self, image):
        *_, height, width = self.input_shape
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image=image.reshape(1, *image.shape)

        return image


    def preprocess_output(self, outputs):
        x_coord = outputs[0][0]
        y_coord = outputs[0][1]

        return [x_coord, y_coord]