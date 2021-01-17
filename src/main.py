import os
import sys
import time
import cv2
import math

from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from head_pose_estimation import ModelHeadPoseEstimation
from facial_landmarks_detection import ModelFacialLandmarkDetection
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController

import logging as log

log.basicConfig(level=log.DEBUG)

MODEL_PRECISION = 'FP16' #FP16-INT8 / FP16 / FP32

FACE_DETECTION_MODEL_FILE = "../models/intel/face-detection-adas-0001/{}/face-detection-adas-0001.xml".format(MODEL_PRECISION)

HEAD_POSE_ESTIMATION_MODEL_FILE = "../models/intel/head-pose-estimation-adas-0001/{}/head-pose-estimation-adas-0001.xml".format(MODEL_PRECISION)

LANDMARKS_REGRESSION_MODEL_FILE = "../models/intel/landmarks-regression-retail-0009/{}/landmarks-regression-retail-0009.xml".format(MODEL_PRECISION)

FACIAL_LANDMARKS_DETECTION_MODEL_FILE = "../models/intel/landmarks-regression-retail-0009/{}/landmarks-regression-retail-0009.xml".format(MODEL_PRECISION)

GAZE_ESTIMATION_MODEL_FILE = "../models/intel/gaze-estimation-adas-0002/{}/gaze-estimation-adas-0002.xml".format(MODEL_PRECISION)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        default='cam',
                        help="path to video file or 'cam' for live feed")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-s", "--show", type=bool, default=False,
                        help='Display video image')
    parser.add_argument("--face_detection_model", required=False, type=str,
                        default=FACE_DETECTION_MODEL_FILE,
                        help="path to Face detection model file")
    parser.add_argument("--headpose_estimation_model", required=False, type=str,
                        default=HEAD_POSE_ESTIMATION_MODEL_FILE,
                        help="path to head pose estimation model file")
    parser.add_argument("--landmarks_regression_model", required=False, type=str,
                        default=LANDMARKS_REGRESSION_MODEL_FILE,
                        help="path to landmarks regression model file")
    parser.add_argument("--gaze_estimation_model", required=False, type=str,
                        default=GAZE_ESTIMATION_MODEL_FILE,
                        help="path to gaze estimation model file")
    return parser

def main():
    args = build_argparser().parse_args()

    log.debug(args)

    # Load face detection model
    faceDetection = ModelFaceDetection(args.face_detection_model, args.prob_threshold, args.device, args.cpu_extension)
    start_model_load_time = time.time()
    faceDetection.load_model()
    facedetection_model_load_time = time.time() - start_model_load_time

    log.debug('Facedetection model load time. {}'.format(facedetection_model_load_time))

    #Load Head pose estimation model
    headPoseEstimation = ModelHeadPoseEstimation(args.headpose_estimation_model, args.prob_threshold, args.device, args.cpu_extension)
    start_model_load_time = time.time()
    headPoseEstimation.load_model()
    headposeestimation_model_load_time = time.time() - start_model_load_time

    log.debug('Head pose estimation model load time. {}'.format(headposeestimation_model_load_time))

    #Facial landmark model
    facialLandmarkDetection = ModelFacialLandmarkDetection(args.landmarks_regression_model, args.prob_threshold, args.device, args.cpu_extension)
    start_model_load_time = time.time()
    facialLandmarkDetection.load_model()
    facialLandmarkDetection_model_load_time = time.time() - start_model_load_time

    log.debug('Facial landmarks detection model load time. {}'.format(facialLandmarkDetection_model_load_time))

    #Gaze estimation model
    gazeEstimation = ModelGazeEstimation(args.gaze_estimation_model, args.prob_threshold, args.device, args.cpu_extension)
    start_model_load_time = time.time()
    gazeEstimation.load_model()
    gazeEstimation_model_load_time = time.time() - start_model_load_time

    log.debug('Gaze estimation model load time. {}'.format(gazeEstimation_model_load_time))

    # Feeder
    feeder = InputFeeder(args.input)
    feeder.load_data()

    counter = 0
    window_name = 'frame'

    facedetection_inference_time_sum = 0
    headpose_inference_time_sum = 0 
    faciallandmark_inference_time_sum = 0
    gazeestimation_inference_time_sum = 0

    #Process Framea
    for frame in feeder.next_batch():
        if frame is None:
            break

        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            break

        #Face detection 
        start_inference_time=time.time()
        face_image, face_coords = faceDetection.predict(frame)
        facedetection_inference_time = time.time() - start_inference_time
        facedetection_inference_time_sum += facedetection_inference_time

        #Head pose estimation
        start_inference_time=time.time()
        yaw, pitch, roll = headPoseEstimation.predict(face_image)
        headpose_inference_time = time.time() - start_inference_time
        headpose_inference_time_sum += headpose_inference_time

        # log.debug('Head pose yaw, pirch ,roll {}, {}, {}'.format(yaw, pitch, roll))

        #Facial landmarks detection
        start_inference_time=time.time()
        left_eye_image, right_eye_image, eye_coords = facialLandmarkDetection.predict(face_image)
        faciallandmark_inference_time = time.time() - start_inference_time
        faciallandmark_inference_time_sum += faciallandmark_inference_time

        # cv2.imwrite('left_eye.png', left_eye_image)
        # cv2.imwrite('right_eye.png', right_eye_image)
        # cv2.imwrite('face.png', face_image) 

        #Gaze estimation
        start_inference_time=time.time()
        gaze_vector = gazeEstimation.predict(left_eye_image, right_eye_image, [yaw, pitch, roll])
        gazeestimation_inference_time = time.time() - start_inference_time
        gazeestimation_inference_time_sum += gazeestimation_inference_time

        #log.debug('Gaze Vector {}, {}'.format(gaze_vector[0], gaze_vector[1]))

        #Mouse
        if(counter%2 ==0):
            mouse = MouseController('high', 'fast')
            mouse.move(gaze_vector[0], gaze_vector[1])

        #Display frame
        if(args.show):
            font = cv2.FONT_HERSHEY_SIMPLEX 
            
            if 0 < len(face_coords):
                #face rect
                fxmin = face_coords[0][0]
                fymin = face_coords[0][1]
                fxmax = face_coords[0][2]
                fymax = face_coords[0][3]

                cv2.rectangle(frame, (fxmin, fymin), (fxmax, fymax), (200,0,0), 2)

                #eye rect
                cv2.rectangle(frame, (fxmin + eye_coords[0][0], fymin + eye_coords[0][1]), (fxmin + eye_coords[0][2], fymin +  eye_coords[0][3]), (0,200,0), 2)
                cv2.rectangle(frame, (fxmin + eye_coords[1][0], fymin + eye_coords[1][1]), (fxmin + eye_coords[1][2], fymin + eye_coords[1][3]), (0,200,0), 2)

                #Face position
                length = 100
                yaw = math.radians(yaw)
                pitch = math.radians(-pitch)
                roll = math.radians(roll)
                x1 = int(length * (math.cos(yaw) * math.cos(roll)))
                y1 = int(length * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)))
                
                x2 = int(length * (-math.cos(yaw) * math.sin(roll)))
                y2 = int(length * (math.cos(pitch) * math.cos(roll) + math.sin(pitch) * math.sin(yaw) * math.sin(roll)))

                x3 = int(length * (math.sin(yaw)))
                y3 = int(length * (-math.cos(yaw) * math.sin(pitch)))

                cv2.line(frame, (fxmin, fymin), (fxmin+x1, fymin+y1), (0,255,0), 2 )
                cv2.line(frame, (fxmin, fymin), (fxmin+x2, fymin+y2), (255,0,0), 2 )
                cv2.line(frame, (fxmin, fymin), (fxmin+x3, fymin+y3), (0,0,255), 2 )

                #gaze 
                x = int(length * gaze_vector[0])
                y = -int(length * gaze_vector[1])

                cv2.line(frame, (fxmax, fymax), (fxmax+x, fymax+y), (0,255,255), 5 )

            else:
                cv2.putText(frame, 'Face not detected', (10, 10), font, 1, (255, 255, 255), 1)

            cv2.imshow(window_name, cv2.resize(frame, (int(frame.shape[1]/3), int(frame.shape[0]/3))))
            


        counter += 1

    log.debug("Face detection inference time average {}".format(facedetection_inference_time_sum/counter))
    log.debug("Headpose inference time average  {}".format(headpose_inference_time_sum/counter))
    log.debug("Faciallandmark inference time average {}".format(faciallandmark_inference_time_sum/counter))
    log.debug("Gazeestimation inference time average {}".format(gazeestimation_inference_time_sum/counter))

    if(args.show):   
        cv2.destroyWindow(window_name)



if __name__ == '__main__':
    main()
    exit(0)