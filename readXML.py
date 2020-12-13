import xml.etree.ElementTree as ET 

import cv2



def plotBndbox(img, archivoXML, fill):
    """Recibe como parametros <imagen> y <archivo xml> 
    Devuelve  imagen con bounding box dibujados"""
    #encuentra la raiz del xml
    raiz =  archivoXML.getroot()
    #busca todos los 'object'
    hijos = raiz.findall('object')
    #set para rellenar bndbox
    thickness = -1
    #itera sobre los objetos extrayendo las dimensiones
    for i in hijos:
        if i.find('name').text == 'planta':
            xmin = int(i.find('bndbox').find('xmin').text)
            ymin = int(i.find('bndbox').find('ymin').text)
            xmax = int(i.find('bndbox').find('xmax').text)
            ymax = int(i.find('bndbox').find('ymax').text)

            # asigna valores a inicio, ancho y largo
            x = xmin 
            y = ymin 
            width = xmax - xmin
            height = ymax - ymin
            #dibuja bounding box sobre la imagen
            if fill == False:
                cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img,(xmin,ymin),(xmin+width,ymin+height), (255,255,255),thickness)
    return img


