#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import cv2
import tensorflow.keras as keras 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from imutils.video import VideoStream
import time


# In[ ]:


print('Loading model...')
model = load_model('mask_detector_2.model')
print('Model loaded...')
print('Starting webcam...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    img = cv2.resize(frame, (224,224), interpolation=cv2.INTER_AREA)
    img = preprocess_input(img)
    img = np.expand_dims(img, 0)
    res = model.predict(img)
    mask = False
    if(res[0][0] > res[0][1]):
        mask = True 
    confidence = res[0][0]
    text = "Mask confidence: {:.2f}%".format(confidence * 100)
    cv2.putText(frame, text, (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Stream", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
vs.stop()
cv2.destroyAllWindows()

