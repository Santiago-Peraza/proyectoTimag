from readXML import plotBndbox
import xml.etree.ElementTree as ET 
import cv2

img = cv2.imread('datos/2018-08-02/DSC_0001.JPG')

archivoXML = ET.parse('datos/2018-08-02/DSC_0001.xml')





bnd = plotBndbox(img,archivoXML)

cv2.namedWindow("ventana", cv2.WINDOW_NORMAL)       
cv2.resizeWindow("ventana", 800, 600)
cv2.imshow('ventana',bnd)
cv2.waitKey(0)
cv2.destroyAllWindows()