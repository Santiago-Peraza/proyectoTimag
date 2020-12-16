import xml.etree.ElementTree as ET 

import cv2
import matplotlib.pyplot as plt

file = ['datos/2018-07-21-23/DSC_0076.JPG','datos/2018-08-02/DSC_0347.JPG', 'datos/2018-07-21-23/DSC_0103.JPG','datos/2018-08-02/DSC_0083.JPG' ]

xml = ['datos/2018-07-21-23/DSC_0076.xml','datos/2018-08-02/DSC_0347.xml', 'datos/2018-07-21-23/DSC_0103.xml','datos/2018-08-02/DSC_0083.xml' ]

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


for i in zip(file,xml):
    imgO =cv2.imread(i[0])
 
    #archivo xml
    archivoXML = ET.parse(i[1])

    img = plotBndbox(imgO,archivoXML,False)
    
    
    cv2.imwrite('example/'+str(i[0].split('/')[2].split('.')[0])+'xml.png', img)
    

