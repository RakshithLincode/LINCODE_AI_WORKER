import torch
from mask_module_utils import detector_get_inference
import pickle
import numpy as np
import cv2 as cv
import cv2 
import matplotlib.pyplot as plt
import glob
import os
from common_utils import *
import argparse
import io
from PIL import Image
import cv2
import torch
import requests
from flask import Flask, request
import base64
from io import BytesIO
import time
import json
from flask import jsonify
from flask import Response
import numpy as np
# from mongohelper import MongoHelper
from base64 import decodestring
import pandas as pd
import os
from PIL import Image
import datetime
from json import dumps
import copy
import bson


rch = CacheHelper()

def mask_get_inference(imo):
    weights = './AI_WEIGHTS/mask_label.pt'
    binary_mask = detector_get_inference(imo,weights)
    return binary_mask

def predict():
    while 1:
        mp = MongoHelper().getCollection("current_inspection")
        # mp_threshold = MongoHelper().getCollection('alignment_threshold')
        # mp_threshold_data = mp_threshold.find_one()
        data = mp.find_one()
        try:
            current_inspection_id = data.get('current_inspection_id')
            print(current_inspection_id)

            if current_inspection_id is None:
                continue
        except:
            pass
        # for i in glob.glob(r'D:\DATA\train\*.jpg'):
        cam = cv2.VideoCapture(0)
        while True:
            ret_val, input_frame = cam.read()
        # input_frame = cv2.imread(i)
            print(input_frame)
            # CacheHelper().set_json({'inspection_trigger':False})
            trigger = CacheHelper().get_json('inspection_trigger')
            rch.set_json({'input_frame':input_frame})
            print(trigger)
            if trigger == True:
                worker_start = time.time()
                select_model = CacheHelper().get_json('current_part_name')
                # select_model = 'M1'
                print(select_model,'select_model..................................................................')
                ocr_frame = copy.copy(input_frame)
                barcode_frame = copy.copy(input_frame)
                frame_input = copy.copy(input_frame)
                result = mask_get_inference(input_frame)
                array = np.array(result)
                c,img_w, img_h = array.shape
                unified_masks = np.zeros((img_w, img_h))
                for mask in array:
                    unified_masks += mask
                unified_masks = unified_masks.astype(np.int32)
                plt.imsave('0.png', unified_masks,cmap='gray')
                masks = cv2.imread('0.png')
                mask ,img_mes, angle_est , value = hemlock_predictor.measurment(masks,frame_input)
                predicted_frame, detector_predections, coordinates = hemlock_predictor.run_inference_hub(lincode_model_2, input_frame)
                ocr_labels = hemlock_predictor.get_ocr_labels(detector_predections)
                ocr_dict = {}
                barcode_dict = {}
                value_measure = []
                for cord_dict in coordinates:
                        for label, cord in cord_dict.items():
                            if label == 'Sealer':
                                xmin = cord[0]
                                ymin = cord[3]
                                xmax = cord[2]
                                ymax = cord[3]
                                tr  = [value[0] , value[1]]
                                tl  = [[cord[0],cord[3]],[cord[2],cord[3]]]
                                mask_frame , measure_frame , measure  = hemlock_predictor.centroid_to_label(mask,img_mes,tl ,tr)
                                value_measure.append(measure)
                for ocr_label in ocr_labels:
                    for cord_dict in coordinates:
                        for label, cord in cord_dict.items():
                            if label == ocr_label:
                                xmin = cord[0]
                                ymin = cord[1]
                                xmax = cord[2]
                                ymax = cord[3]
                                crop_img = ocr_frame[ymin:ymax,xmin:xmax]
                                ocr_result = hemlock_predictor.get_ocr_results(crop_img)
                                ocr_dict[label] = ocr_result
                print(ocr_dict,'OCR_dict values')
                barcode_dict = hemlock_predictor.barcode_value_decoder(barcode_frame)
                print(barcode_dict,'Barcode_dict values')
                print(detector_predections,ocr_dict,barcode_dict,angle_est,value_measure,select_model,'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
                response = hemlock_predictor.check_kanban(detector_predections,ocr_dict,barcode_dict,angle_est,value_measure,select_model)
                print(response)
                response['part_name'] = select_model
                # wr = MongoHelper().getCollection(str(current_inspection_id)+"_worker_results")
                # # wr = MongoHelper().getCollection(current_inspection_id+"worker_results")
                # wr.insert_one(response)
                is_accepted = response['status'] 
                x = bson.ObjectId()
                cv2.imwrite(datadrive_path+str(x)+'_ip.jpg',input_frame)
                cv2.imwrite(datadrive_path+str(x)+'_mask.jpg',mask_frame)
                cv2.imwrite(datadrive_path+str(x)+'_measure.jpg',measure_frame)
                cv2.imwrite(datadrive_path+str(x)+'_pf.jpg',predicted_frame)

                input_frame_path = 'http://localhost:3306/'+str(x)+'_ip.jpg'
                mask_frame_path = 'http://localhost:3306/'+str(x)+'_mask.jpg'
                measure_frame_path = 'http://localhost:3306/'+str(x)+'_measure.jpg'
                predicted_frame_path = 'http://localhost:3306/'+str(x)+'_pf.jpg'

                print(input_frame_path)
                rch.set_json({"input_frame_path":input_frame_path})
                rch.set_json({"mask_frame":mask_frame_path})
                rch.set_json({"measure_frame":measure_frame_path})
                rch.set_json({"inference_frame":predicted_frame_path})
                rch.set_json({"status":is_accepted})
                # rch.set_json({"defect_list":conf.defects})
                rch.set_json({"feature_mismatch":response['features']})
                rch.set_json({"defects":response['defects']})
                rch.set_json({"ocr_barcode_mismatch":response['ocr_barcode_mismatch']})
                rch.set_json({"label_angle":response['label_angle']})
                rch.set_json({"label_to_sealent_measurment":response['label_to_sealent_measurment']})
                data = {'current_inspection_id':str(current_inspection_id)}#,'raw_frame':input_frame_path,'inference_frame':inference_frame_path,'status':is_accepted,'defect_list':conf.defects,'feature_list':conf.feature,'features':[],'defects':defect_list}
                requests.post(url = 'http://localhost:8000/livis/v1/inspection/save_inspection_details/', data = data)
                CacheHelper().set_json({'inspection_trigger':False})
                print("Worker_Time_Taken",time.time() - worker_start)
                if cv2.waitKey(1) == 27: 
                    break  # esc to quit
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    start = time.time()
    # parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()
    hemlock_predictor = Hemlock_Process()
    datadrive_path = 'D:/BACKEND_HEMLOCK/Hemlock_Backend/datadrive/'
    lincode_model_2 = hemlock_predictor.load_model()
    warmup_image = cv2.imread(r'D:\mmdetection\mmdetection\runs\detect\exp2\image0.jpg') # Warmup Load
    predicted_frame, detector_predections, coordinates = hemlock_predictor.run_inference_hub(lincode_model_2, warmup_image)
    warm_up_result = mask_get_inference(warmup_image)
    print("load architecture",time.time() - start)
    predict()

    
