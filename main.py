from find_plant import *
from etiquetar import *
import cv2
import numpy as np
from readXML import plotBndbox
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt
from bndbox import bndbox
from cropped import cropped
import matplotlib.gridspec as gridspec


# 07/21-23
path = 'datos/2018-07-21-23/'

#nombre de la imagen
name = 'DSC_0419.JPG'
imgO =cv2.imread(path+name)
 
#archivo xml
archivoXML = ET.parse(path+'DSC_0419.xml')

testBox = imgO.copy()
image = imgO.copy()

# 07/30
# imgO =cv2.imread('datos/2018-07-30/DSC_0050.JPG')
# archivoXML = ET.parse('datos/2018-07-21-23/DSC_0076.xml')

# 08/02
# imgO =cv2.imread('datos/2018-08-02/DSC_0097.JPG')
# archivoXML = ET.parse('datos/2018-08-02/DSC_0097.xml')

# Casos particulares
# casosParticulares = ['datos/2018-07-21-23/DSC_0001.JPG','datos/2018-07-21-23/DSC_0036.JPG','datos/2018-07-21-23/DSC_0103.JPG','datos/2018-07-21-23/DSC_0157.JPG','datos/2018-08-02/DSC_0062.JPG','datos/2018-08-02/DSC_0083.JPG','datos/2018-07-21-23/DSC_0331.JPG','datos/2018-07-21-23/DSC_0352.JPG']
# xmlCasosParticulares = ['datos/2018-07-21-23/DSC_0001.xml','datos/2018-07-21-23/DSC_0036.xml','datos/2018-07-21-23/DSC_0103.xml','datos/2018-07-21-23/DSC_0157.xml','datos/2018-08-02/DSC_0062.xml','datos/2018-08-02/DSC_0083.xml','datos/2018-07-21-23/DSC_0331.xml','datos/2018-07-21-23/DSC_0352.xml']


# Imagenes buenas
# buenas = ['datos/2018-07-21-23/DSC_0017.JPG','datos/2018-07-21-23/DSC_0286.JPG','datos/2018-07-21-23/DSC_0309.JPG','datos/2018-07-21-23/DSC_0401.JPG','datos/2018-07-21-23/DSC_0437.JPG', 'datos/2018-08-02/DSC_0010.JPG','datos/2018-08-02/DSC_0105.JPG','datos/2018-08-02/DSC_0106.JPG','datos/2018-08-02/DSC_0190.JPG','datos/2018-08-02/DSC_0208.JPG','datos/2018-08-02/DSC_0276.JPG','datos/2018-08-02/DSC_0347.JPG']
# xmlBuenas = ['datos/2018-07-21-23/DSC_0017.xml','datos/2018-07-21-23/DSC_0286.xml','datos/2018-07-21-23/DSC_0309.xml','datos/2018-07-21-23/DSC_0401.xml','datos/2018-07-21-23/DSC_0437.xml', 'datos/2018-08-02/DSC_0010.xml','datos/2018-08-02/DSC_0105.xml','datos/2018-08-02/DSC_0106.xml','datos/2018-08-02/DSC_0190.xml','datos/2018-08-02/DSC_0208.xml','datos/2018-08-02/DSC_0276.xml','datos/2018-08-02/DSC_0347.xml']

#umbrales para color
lowGreen_1 = np.array([43,60,0], np.uint8)
highGreen_1 = np.array([80,255,255], np.uint8)

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

#dilatacion por segunda vez para mejorar etiquetado
segundaDilatacion =  cv2.dilate(dilate,kernel,iterations=3)
#obtengo etiquetado
_,segundaEtiqueta = cv2.connectedComponents(segundaDilatacion, connectivity=8)
# obtengo etiquetado
segundaLabel = cv2.bitwise_and(segundaEtiqueta,segundaEtiqueta, mask=dilate)


###################################
# genero bounding box sogre la imagen testBox y almaceno en bndAutomatico

# si fill == True rellena los rectangulos
bndAutomatico = bndbox(segundaLabel, testBox, False)

###################################



fig, (ax1,ax2) = plt.subplots(1,2,sharey = True,sharex = True)
ax1.set_title('Bounding box automaticos')
ax1.imshow(bndAutomatico)
ax2.set_title('Original')
ax2.imshow(segundaLabel)

plt.show()

cv2.imwrite('./example/'+name+'_boundingbox.jpg', bndAutomatico)
# cv2.imwrite('./example/segmentation.jpg', andMask)
# cv2.imwrite('./example/contours.jpg', img)

bndboxByXML = np.zeros(image.shape,  np.uint8)
bndboxByXML = plotBndbox(bndboxByXML,archivoXML,True)
bndboxByXML = cv2.cvtColor(bndboxByXML,cv2.COLOR_RGB2GRAY)

xmin,ymin,xmax,ymax = cropped(bndboxByXML)

bndboxByXML = bndboxByXML[ymin:ymax,xmin:xmax]
bndboxByColor = np.zeros(image.shape,  np.uint8)
bndboxByColor = bndbox(segundaLabel, bndboxByColor,True)
bndboxByColor = cv2.cvtColor(bndboxByColor,cv2.COLOR_RGB2GRAY)
bndboxByColor = bndboxByColor[ymin:ymax,xmin:xmax]

andbndbox  = cv2.bitwise_and(bndboxByColor,bndboxByXML)
orbndbox = cv2.bitwise_or(bndboxByXML,bndboxByXML)

verdaderoPositivo = andbndbox

falsoPositivo= cv2.bitwise_and(cv2.bitwise_not(bndboxByXML), bndboxByColor)

falsoNegativo = cv2.bitwise_and(bndboxByXML, cv2.bitwise_not(bndboxByColor))

verdaderoNegativo = cv2.bitwise_and(cv2.bitwise_not(bndboxByXML) , cv2.bitwise_not(bndboxByColor))

sensibilidad = np.count_nonzero(verdaderoPositivo)/(np.count_nonzero(verdaderoPositivo)+ np.count_nonzero(falsoNegativo))

especificidad = np.count_nonzero(verdaderoNegativo)/(np.count_nonzero(verdaderoNegativo)+ np.count_nonzero(falsoPositivo))

fig= plt.figure(figsize=(15, 15))
G = gridspec.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace= 0.3)

ax1 = plt.subplot(G[0,0])
plt.imshow(imgO, 'gray')
plt.setp(ax1, title=u'Imágen parsela')
plt.axis("off")

ax1 = plt.subplot(G[0,1])
plt.imshow(verdaderoPositivo, 'gray')
plt.setp(ax1, title=u'Verdaderos positivos')
plt.axis("off")

ax2 = plt.subplot(G[0,2])
plt.imshow(verdaderoNegativo, 'gray')
plt.setp(ax2, title=u'Verdaderos negativos')
plt.axis("off")

ax3= plt.subplot(G[1,0])
plt.imshow(falsoPositivo, 'gray')
plt.setp(ax3, title=u'Falsos positivos')
plt.axis("off")

ax4 = plt.subplot(G[1,1])
plt.imshow(falsoNegativo, 'gray')
plt.setp(ax4, title=u'Falsos negativos')
plt.axis("off")

ax5 = plt.subplot(G[1,2])
ax5.text(0.3,0.5, "Sensibilidad = {:.2f} ".format(sensibilidad,2))
ax5.text(0.3,0.4, "Especificidad = {:.2f} ".format(especificidad,2))
plt.axis("off")

plt.draw()


