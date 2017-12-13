import urllib.request
import cv2
import numpy as np
import os

def store_raw_images():
    images_link = 'URL to retrieve images from'   
    image_urls = urllib.request.urlopen(images_link).read().decode()
    pic_num = 1
    
    if not os.path.exists('destination directory'):
        os.makedirs('destination directory')
        
    for i in image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "destination directory/"+str(pic_num)+".jpeg")
            #img = cv2.imread("scrapped/"+str(pic_num)+".jpg")
            #cv2.imwrite("scrapped/"+"weed."+str(pic_num)+".jpeg",img)
            pic_num += 1
            
        except Exception as e:
            print(str(e))  

store_raw_images()