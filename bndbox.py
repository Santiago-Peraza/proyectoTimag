import numpy as np
import cv2

def bndbox(labelList, img):
    labels = np.delete(np.unique(labelList),0)
    
    
    for label in labels:
        
        position = np.argwhere(labelList==label)

        ymin = position[:,0].min() 
        xmin = position[:,1].min() 

        ymax = position[:,0].max() 
        xmax = position[:,1].max() 
        height = ymax - ymin
        width = xmax - xmin
        # print(label, xmin, ymin, width, height)
        cv2.rectangle(img,(xmin,ymin),(xmin+width,ymin+height), (255,0,0), 2, cv2.LINE_AA)
    return img
