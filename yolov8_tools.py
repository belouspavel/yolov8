from ultralytics import YOLO
import numpy as np
import pandas as pd
from IPython.display import clear_output
import cv2 
import os
from sklearn.metrics import precision_recall_fscore_support




def get_boxes(result):
  orig_shp = result[0].orig_shape
  all_boxes = np.empty((0, 7))
  for i in range(len(result)):
    bbox = result[i].cpu().boxes.data.numpy()
    bbox = np.hstack((bbox, np.tile(i, (bbox.shape[0], 1))))
    all_boxes = np.vstack((all_boxes, bbox))
  return result, all_boxes, orig_shp

def detect_videos(path_model, model_in_path, video_source):
  length = len([f for f in os.listdir(video_source) 
     if f.endswith('.mp4') and os.path.isfile(os.path.join(video_source, f))]) # подсчитаем количество видео в папке
  
  for N in range(42,length+1): # устанавливаем какие видео смотрим
    try:
        with open(video_source + f'{N}.mp4', 'r') as f:
          model = YOLO(path_model+model_in_path)  ## каждый раз инициализируем модель в колабе иначе выдает ошибочный результат
          results, all_boxes, orig_shape = get_boxes(model.predict(video_source + f'{N}.mp4',
                                                                line_thickness = 2,vid_stride = 1, save = True))
          np.save(path_model + f"{N}.npy", np.array((orig_shape, all_boxes)))
    except:    
        print(f'Видео {N}: отсутствует')