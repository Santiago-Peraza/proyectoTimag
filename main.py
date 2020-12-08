from find_plant import *
from etiquetar import *
import cv2
import numpy as np
from readXML import plotBndbox
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt

from bndbox import bndbox
# 07/21-23
imgO =cv2.imread('datos/2018-07-21-23/DSC_0076.JPG')
archivoXML = ET.parse('datos/2018-07-21-23/DSC_0076.xml')

testBox = imgO.copy()
image = imgO.copy()

# 07/30
# imgO =cv2.imread('datos/2018-07-30/DSC_0050.JPG')
# archivoXML = ET.parse('datos/2018-07-21-23/DSC_0076.xml')

# 08/02
# imgO =cv2.imread('datos/2018-08-02/DSC_0097.JPG')
# archivoXML = ET.parse('datos/2018-08-02/DSC_0097.xml')

# Casos particulares
# casosParticulares = ['datos/2018-07-21-23/DSC_0036.JPG','datos/2018-07-21-23/DSC_0103.JPG','datos/2018-07-30/DSC_0122.JPG','datos/2018-07-30/DSC_0411.JPG','datos/2018-08-02/DSC_0062.JPG','datos/2018-08-02/DSC_0038.JPG']


#umbrales para color
lowGreen_1 = np.array([43,60,0], np.uint8)
highGreen_1 = np.array([75,255,255], np.uint8)

#umbrales para forma
lowGreen_2 = np.array([10,0,0], np.uint8)
highGreen_2 = np.array([110,255,255], np.uint8)

#busco color verde en imagen
color = search_plants(imgO,lowGreen_1,highGreen_1)

#genero kernel  para dilatacion
kernel = np.ones((7,7), np.uint8)
kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)) 

#dilatacion en filtrado por color
color = cv2.dilate(color,kernel,iterations=1)

# busco plantas ampliando rango de color
tamaño = search_plants(imgO,lowGreen_2,highGreen_2)

# realizo AND entre mascaras anteriores
andMask = cv2.bitwise_and(color,tamaño)


# dilate = cv2.medianBlur(res,9)
dilate = cv2.morphologyEx(andMask, cv2.MORPH_OPEN, kernel)

# colorizo deteecciones 
and_delate = cv2.bitwise_and(image,image, mask=dilate)


segundaDilatacion =  cv2.dilate(dilate,kernel,iterations=3)
_,segundaEtiqueta = cv2.connectedComponents(segundaDilatacion, connectivity=8)
segundaLabel = cv2.bitwise_and(segundaEtiqueta,segundaEtiqueta, mask=dilate)


###################################
bndAutomatico = bndbox(segundaLabel, testBox)



###################################

# cv2.namedWindow("original", cv2.WINDOW_NORMAL)       
# cv2.resizeWindow("original", 800, 600)
# cv2.imshow('original',segundaLabel)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



fig, (ax1,ax2) = plt.subplots(1,2,sharey = True,sharex = True)
ax1.set_title('Bounding box automaticos')
ax1.imshow(bndAutomatico)
ax2.set_title('Original')
ax2.imshow(segundaLabel)

plt.show()

cv2.imwrite('./example/boundingbox.jpg', bndAutomatico)
# cv2.imwrite('./example/segmentation.jpg', andMask)
# cv2.imwrite('./example/contours.jpg', img)
