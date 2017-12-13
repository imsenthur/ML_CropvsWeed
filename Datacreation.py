import os
import shutil
from tqdm import tqdm
#Reads images from a local folder then renames it which helps preprocessing the data.
def navigate_and_rename(pic_num,src,dist):

    for item in tqdm(os.listdir(src)):
        print(pic_num)
        print(item)
        s = os.path.join(src, item)
        shutil.copy(s, os.path.join(dist,"crop."+str(pic_num)+".jpeg"))   
        pic_num += 1 

pic_num = 1
dir_src = "/source directory/"
dist = "/destination directory/"
navigate_and_rename(pic_num,dir_src,dist)