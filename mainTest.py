import cv2
from keras.models import load_model
from PIL import Image 
import numpy as np 

#loading our model
model=load_model('BrainTumor10EpochsCategorical.h5')


#loading images now 
image=cv2.imread('C:\\Users\\suhan\\OneDrive\\Desktop\ML_PROJ\\Brain_tumor_proj\\archive (6)\\pred\\Y14.jpg')

#convert these images to array
img=Image.fromarray(image)
img=img.resize((64,64))
#converting theseimages to numpy array
img=np.array(img)

input_img=np.expand_dims(img,axis=0)    #on running this , this shows o/p=0 which is no tumor 




result=model.predict_classes(input_img)
print(result)







#Nnote-1) jb hum images ko test krr rhe h and o/p agr sab 0 aarha h jaise  for image- no 100 then all array coming 0 that is no tumor otherwise y71 use lri toh array mei number aarhe h 1,2, so on which shows tumor is there 
 
