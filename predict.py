from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os

# Test görüntüsünü yüklüyoruz.
imagePath = r"datasetSplit/test/Benigns/16_1.jpg"
image = cv2.imread(imagePath)
output = image.copy()

# Sınıflandırma için görüntüyü ön işleme alıyoruz.
image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Eğittiğimiz CNN ağını yüklüyorz.
print("[INFO] loading network...")
model = load_model("thyroid2.model")

# Sınıf listemizi oluşturuyoruz.
classes = ["Benigns","Maligns"]

# Görüntümüzü sınıflandırıyoruz.
print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = classes[idx]




# Seçilen görüntünün dosya yolundaki klasör etiketine göre modelimizin tahmin
# ettiği etiketi karşılaştırıp doğruysa "correct" yazdırıyoruz.
filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
print("filename:",filename)
print("sonuc:",filename.rfind(label))
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# Etiketi oluşturuyoruz ve etiketi resmin üzerine çiziyoruz
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# Sonucu görüntülüyoruz.
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)

