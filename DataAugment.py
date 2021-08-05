import glob
import random
import cv2
import numpy as np

# Bu fonksiyonda görüntüyü dikeyde ve yatayda dödürme işlemi
# gerçekleştireceğiz.

def flip(path, vflip=False, hflip=False):
    '''
    Flip the image
    :param image: image to be processed
    :param vflip: whether to flip the image vertically
    :param hflip: whether to flip the image horizontally
    '''
    img = cv2.imread(path)
    if hflip or vflip:
        if hflip and vflip:
            path = path.replace(".jpg", "_hvf.jpg")
            c = -1
        else:
            c = 0 if vflip else 1
        img = cv2.flip(img, flipCode=c)
        if vflip == True and hflip == False:
            path = path.replace(".jpg", "_vf.jpg")
        if hflip == True and vflip == False:
            path = path.replace(".jpg", "_hf.jpg")
        cv2.imwrite(path, img)
# Bu fonksiyonda görüntüye random bir parlaklık değeri vereceğiz.
def brightness_augment(path):
    factor = round(random.uniform(0.2,0.7),1)
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    path = path.replace(".jpg","_br.jpg")
    cv2.imwrite(path,rgb)
# Bening de bulunan veriler daha az olduğu için veri büyütme
# işelmini bunun üzerinde gerçekleştireceğiz.
for path in glob.glob("dataset/Benigns/*.jpg"):
    flip(path,hflip=True)
    flip(path, vflip=True)
    flip(path, vflip=True, hflip=True)
    brightness_augment(path)

