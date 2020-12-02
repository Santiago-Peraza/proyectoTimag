def search_plants(img,umbL, umbU):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    imgF = cv2.bitwise_and(img,img, mask= mask)
    return imgF