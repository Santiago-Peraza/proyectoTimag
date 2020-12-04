

mask = cv2.inRange(img,lowGreen,highGreen)


newimg = cv2.bitwise_and(img,img,mask=mask)

bnd = plotBndbox(newimg,archivoXML)

cv2.namedWindow("ventana", cv2.WINDOW_NORMAL)       
cv2.resizeWindow("ventana", 800, 600)
cv2.imshow('ventana',mask)

cv2.namedWindow("original", cv2.WINDOW_NORMAL)       
cv2.resizeWindow("original", 800, 600)
cv2.imshow('original',imgO)
cv2.waitKey(0)
cv2.destroyAllWindows()