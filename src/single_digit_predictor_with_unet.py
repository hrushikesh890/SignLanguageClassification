import cv2
import numpy as np
import keras 
import keras.backend as TF
import time
from keras.models import load_model
import tensorflow as tf
from keras.utils.generic_utils import CustomObjectScope

def dice_coef(y_true, y_pred):
    y_true_f = TF.flatten(y_true)
    y_pred_f = TF.flatten(y_pred)
    intersection = TF.sum(y_true_f * y_pred_f)
    return (2. * intersection + TF.epsilon()) / (TF.sum(y_true_f) + TF.sum(y_pred_f) + TF.epsilon())

img_width = 224
img_height = 224
channels = 3

vidcap = cv2.VideoCapture(0)
count=0
success,fimage = vidcap.read()
img_size = (224,224)
font = cv2.FONT_HERSHEY_SIMPLEX

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

for c in range(0,20):
	success,fimage = vidcap.read()
cv2.imwrite("fram%d.jpg" % count, fimage)
fimage = cv2.cvtColor(fimage, cv2.COLOR_BGR2GRAY)


for c in range(30,0,-1):
	success,fimage = vidcap.read()
	cv2.waitKey(50)
	#cv2.imshow('Gesture', fimage)
	
	#print(c)
	cv2.putText(fimage, 'Do the Gesture', (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.putText(fimage, str(c), (50, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
	cv2.imshow('Gesture', fimage)
print('done')

model_hand = load_model('my_color_model.hdf5',custom_objects={'dice_coef':dice_coef})

test1 = cv2.resize(fimage, dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)
test2=test1
test1 = (test1 - np.min(test1)) / (np.max(test1) - np.min(test1))
test = test1
im2 = model_hand.predict(test.reshape(1,img_height, img_width, channels))
im2 = np.squeeze(im2)
im2 = im2*255
ret, im2 = cv2.threshold(im2, 150, 255,0)

kernel = np.ones((7,7),np.uint8)
kernel1 = np.ones((3,3),np.uint8)
im2 = cv2.erode(im2,kernel1,iterations=3)

im2 = cv2.morphologyEx(im2,cv2.MORPH_OPEN, kernel)
im2 = cv2.morphologyEx(im2,cv2.MORPH_CLOSE, kernel)
im2 = cv2.dilate(im2,kernel,iterations=5)
im2 = cv2.convertScaleAbs(im2)
#print(im2.dtype)

im2, contours, hierarchy = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(len(contours))
largest_area = 0
if(len(contours)>0):
  for x in range(len(contours)):
    a = cv2.contourArea(contours[x])
    if(a>=largest_area):
        largest_area = a
        largest_contour_index = x

  c = contours[largest_contour_index]
  #print(c.shape)

  rect = cv2.boundingRect(c)
  #print(cv2.contourArea(c))
  #print((x,y),(x+w,y+h))
  x,y,w,h = rect
  cv2.rectangle(im2,(x,y),(x+w,y+h),(0,255,0),2)

  #print((x,y),(x+w,y+h))
  cropped_image = test2[y:y+h,x:x+w,:]
  cropped_image = cv2.resize(cropped_image, dsize=(img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
  
  #cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image))
  #plt.imshow(im,cmap='gray')
  cv2.rectangle(test,(x,y),(x+w,y+h),(255,0,0),3)
  cv2.imwrite("frame2.jpg" , cropped_image)
  
else:
  print('No Hand detected')

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
	model_predict = load_model('drivev1_2(224,224_10).hdf5')
	#model_predict = load_model('mobile(224,224_1).hdf5')
#model_predict.summary()
cropped_image = cv2.resize(cropped_image,img_size)
img_list=[]
img_list.append(cropped_image)
img_list.append(cropped_image)
img_data = np.array(img_list)
img_data = img_data.astype('float32')
#print(img_data.shape)
img_data = img_data.reshape(img_data.shape)# + (1,))
#print(img_data.shape)
k=model_predict.predict(img_data,verbose=1)
#print(k)
#print((np.argmax(k[0])))
max = (int)(np.max(k)*100)
print('\n\n\n\n\n\n'+str((np.argmax(k[0])))+' detected with ' + str(max) + '% probability\n\n\n\n')
