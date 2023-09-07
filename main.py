import cv2
import easyocr
import matplotlib.pyplot as plt

#read the image
image_path =  r'C:\Users\MONSTER\Desktop\Computer Vision\data\test2.png'

img = cv2.imread(image_path)

#instance text detector
text_reader = easyocr.Reader(["en"],gpu=False) # read english texts

#detect text on the image
text_detections = text_reader.readtext(img)

#draw bounding boxes and the text

threshold = 0.25 #threshold score for text detection

for t in text_detections:

    bbox, text, score = t

    if(score > threshold):

        cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 2)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#plot the test images
plt.imshow(img)
plt.show()