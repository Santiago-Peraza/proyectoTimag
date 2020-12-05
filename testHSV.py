from find_plant import *
from etiquetar import *
import cv2
import numpy as np
from readXML import plotBndbox
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt
# 07/21-23
imgO =cv2.imread('datos/2018-07-21-23/DSC_0076.JPG')
archivoXML = ET.parse('datos/2018-07-21-23/DSC_0076.xml')


image = imgO.copy()

# 07/30
# imgO =cv2.imread('datos/2018-07-30/DSC_0050.JPG')
# archivoXML = ET.parse('datos/2018-07-21-23/DSC_0076.xml')

# 08/02
# imgO =cv2.imread('datos/2018-08-02/DSC_0097.JPG')
# archivoXML = ET.parse('datos/2018-08-02/DSC_0097.xml')


#umbrales para color
lowGreen_1 = np.array([43,70,0], np.uint8)
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



# plt.figure()
# plt.imshow(color)
# plt.figure()
# plt.imshow(tamaño)

# plt.figure()
# plt.imshow(andMask)

#dibujo bounding box en imagen
bnd = plotBndbox(imgO,archivoXML)


# dilate = cv2.medianBlur(res,9)
dilate = cv2.morphologyEx(andMask, cv2.MORPH_OPEN, kernel)

# colorizo deteecciones 
and_delate = cv2.bitwise_and(image,image, mask=dilate)


img = imgO.copy()
# ret, thresh = cv2.threshold(cv2.cvtColor(dilate,cv2.COLOR_HSV2GRAY), 127, 255, 0)
contours,_ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255,0,0), 3)


ret, labels = cv2.connectedComponents(dilate, connectivity=8)
# label_hue = np.uint8(179*labels/np.max(labels))
# blank_ch = 255*np.ones_like(label_hue)
# labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# # Converting cvt to BGR
# labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# # set bg label to black
# labeled_img[label_hue==0] = 0
segundaDilatacion =  cv2.dilate(dilate,kernel,iterations=3)
_,segundaEtiqueta = cv2.connectedComponents(segundaDilatacion, connectivity=8)
segundaLabel = cv2.bitwise_and(segundaEtiqueta.astype(np.uint8),dilate)
# # labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_BGR2HSV)
plt.figure()
plt.title('la cosita del coso')
plt.imshow(segundaLabel)
# plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
prueba = segundaLabel.copy()
contours,_ = cv2.findContours(segundaLabel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(prueba, contours, -1, (0,255,255), 3)


# convex = cv2.convexHull(segundaLabel[segundaLabel==231])
# rect = cv2.minAreaRect(segundaDilatacion[segundaLabel==231])
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(segundaLabel,[box],0,(0,0,255),2)
# cv.drawContours(segundaLabel, convex)

plt.figure()
plt.title('contorno')
plt.imshow(segundaLabel)
plt.show()
# cv2.namedWindow("original", cv2.WINDOW_NORMAL)       
# cv2.resizeWindow("original", 800, 600)
# cv2.imshow('original',dilate)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cv2.imwrite('./example/boundingbox.jpg', bnd)
cv2.imwrite('./example/segmentation.jpg', andMask)
cv2.imwrite('./example/contours.jpg', img)