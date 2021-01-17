'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore
from ModelInferenceBase import ModelInferenceBase

class ModelFaceDetection(ModelInferenceBase):
    '''
    Class for the Face Detection Model.
    '''
       
    def predict(self, image):
        processed_image = self.preprocess_input(image)
        input_dict={self.input_name:processed_image}

        result = self.exec_net.infer(input_dict)[self.output_name]

        image, coords = self.preprocess_output(result, image)

        return image, coords

    def preprocess_input(self, image):
        *_, height, width = self.input_shape
        image = cv2.resize(image, (width, height))
        image = image.transpose((2, 0, 1))
        image=image.reshape(1, *image.shape)

        return image


    def preprocess_output(self, outputs, image):
        coords = []
        
        for obj in outputs[0][0]:
            conf = obj[2]
            if conf >= self.prob_threshold:
                x_min = int(obj[3] * image.shape[1])
                y_min = int(obj[4] * image.shape[0])
                x_max = int(obj[5] * image.shape[1])
                y_max = int(obj[6] * image.shape[0])
                coords.append([x_min, y_min, x_max, y_max])
                image = image[y_min:y_max, x_min:x_max]


        return image, coords