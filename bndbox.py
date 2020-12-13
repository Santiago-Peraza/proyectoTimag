import numpy as np
import cv2

def bndbox(labelList, img, fill):
    labels = np.delete(np.unique(labelList),0)
    thickness = -1
    
    for label in labels:
        
        position = np.argwhere(labelList==label)

        ymin = position[:,0].min() 
        xmin = position[:,1].min() 

        ymax = position[:,0].max() 
        xmax = position[:,1].max() 
        height = ymax - ymin
        width = xmax - xmin
        # si  fill ==True rellena los rectangulos
        if fill == False:
            cv2.rectangle(img,(xmin,ymin),(xmin+width,ymin+height), (0,0,255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(img,(xmin,ymin),(xmin+width,ymin+height), (255,255,255),thickness)
            
    return img
            