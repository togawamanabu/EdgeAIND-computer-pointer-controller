'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
from openvino.inference_engine import IECore

import logging as log

class ModelInferenceBase:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, prob_threshold, device='CPU', extensions=None):
        self.net = None
        self.model_name = model_name
        self.prob_threshold = prob_threshold
        self.device=device
        self.extensions = extensions

    def load_model(self):
        model_xml = self.model_name
        model_bin = os.path.splitext(model_xml)[0] + ".bin" 
        
        self.core = IECore()

        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extentions(cpu_extention, 'CPU')

        self.net = self.core.read_network(model=model_xml, weights=model_bin)

        self.exec_net = self.core.load_network(network=self.net, device_name=self.device)

        self.check_model()

    def predict(self, image):
        pass

    def check_model(self):
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape


    def preprocess_input(self, image):
        pass

    def preprocess_output(self, outputs, image):
        pass