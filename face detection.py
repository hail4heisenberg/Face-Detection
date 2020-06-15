import numpy as np
import cv2 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
test_image = cv2.imread('baby1.jpg')
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
plt.imshow(test_image_gray, cmap='gray')
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


haar_cascade_face = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')


faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 1.2, minNeighbors = 5);
print('Faces found: ', len(faces_rects))
for (x,y,w,h) in faces_rects:
     cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(convertToRGB(test_image))
def detect_faces(cascade, test_image, scaleFactor = 1.1):
    image_copy = test_image.copy()
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors = 5)
    
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)
        
    return image_copy
test_image2 = cv2.imread('group.jpg')
faces = detect_faces(haar_cascade_face, test_image2)
plt.imshow(convertToRGB(faces))
cv2.imwrite('image1.png',faces)
