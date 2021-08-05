import cv2
import glob

# Verisetimizdeki ultrason görüntülerinin dışında bulunan siyah alandan kurtulup sadece ultrason görüntüsünü
# elde etmek için openCV kütüphanesi ile kenar bulma işlemi gerçekleştireceğiz.
for path in glob.glob("dataset2/*.jpg"):
    # Görüntüyü yüklüyoruz, gri tonlamaya dönüştürüp , kenarlarını buluyoruz.
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    
    # Kontur buluyoruz ve kontur alanına göre sıralıyoruz
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Sınırlayıcı kutuyu buluyoruz ve ayıklıyoruz(En dış kenarlığı alıyoruz)
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        break
    ROI = cv2.resize(ROI,(430,360))
    path = path.replace("dataset2","dataset")
    cv2.imwrite(path,ROI)
    cv2.waitKey()