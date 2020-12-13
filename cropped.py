import numpy as np

def cropped(img):
    position = np.argwhere(img!=0)

    ymin = position[:,0].min() 
    xmin = position[:,1].min() 

    ymax = position[:,0].max() 
    xmax = position[:,1].max() 

    return xmin,ymin,xmax,ymax