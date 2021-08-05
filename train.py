# matplotlib arka ucunu ayarlıyoruz, böylece rakamlar arka planda kaydedilebilir.
import matplotlib
matplotlib.use("Agg")

# Gerekli kütüphaneleri dahil ediyoruz.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from createCNN import CreateCNN
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import tensorflow as tf
import random
import pickle
import cv2
import os


# Keras CNN'imizi eğitirken kullanılan önemli değişkenleri tanımlıyoruz.
# Epoch: Ağımızı eğiteceğimiz toplam dönem sayısı (yani, ağımızın her bir eğitim örneğini kaç kez "gördüğü" ve ondan öğrenilen ağırlıklar).
# INIT_LR: İlk öğrenme oranı — 1e-3 değeri, ağı eğitmek için kullanacağımız optimize edici olan Adam optimizer için varsayılan değerdir.
# Batch Size (BS):Eğitim için ağımıza toplu görüntü aktaracağız. Epoch başına birden fazla parti vardır. BS Değeri aynı anda kaç görüntü alınacağını kontrol eder.
# IMAGE_DIMS: Burada girdi görüntülerimizin uzamsal boyutlarını sağlıyoruz. Giriş görüntülerimizin kanallı 430 x 360 pikseller 3(yani RGB) olmasını isteyeceğiz .
EPOCHS = 100
INIT_LR = 1e-3
BS = 1
IMAGE_DIMS = (430, 360, 3)

# Önceden işlenmiş görüntüleri ve etiketleri tutacak veriler ve etiketler olmak üzere iki liste başlatıyoruz.
data = []
labels = []

# Verisetimizdeki tüm görüntü yollarını alıyoruz ve rastgele karıştırıyoruz.
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images("dataset")))
random.seed(42)
random.shuffle(imagePaths)

# Giriş görüntülerimiz üzerinde bir döngü başlatıyoruz.
for imagePath in imagePaths:
	# görüntüyü yüklüyoruz, yeniden boyutlandırıp ve veri listesinde saklayoruz.
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# Sınıf etiketini görüntü yolundan çıkarıyoruz ve etiketler listesini güncelliyoruz.
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
print(labels)

# Ham piksel yoğunluklarını [0, 1] aralığına ölçeklendiriyoruz.
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# Etiketleri ikili hale getiriyoruz.
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print("labels",lb)

# verilerin %80'ini eğitim için ve kalan %20'sini doğrulama için kullanarak
# verileri eğitim ve doğrulama bölümlerine ayırıyoruz.

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# veri zengişleştirme için ImageDataGenerator oluşturuyoruz. Burada rastgele döndürme,
# rastgele kaydırma, rastgele yakınlaşma gibi çeşitli işlemlere tabi tutuyoruz eğitim verilerimizi
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# Modeli oluşturup başlatıyoruz.
print("[INFO] compiling model...")
model = CreateCNN.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])


# Eğitim ağını başlatıyoruz.
print("[INFO] training network...")
H = model.fit(trainX, trainY,
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# Model ağırlığını belirtilen(istenilen) dizine kaydediyoruz.
print("[INFO] serializing network...")
model.save("thyroid.model", save_format="h5")

# Etiket dosyamızıda kaydediyoruz.
print("[INFO] serializing label binarizer...")
f = open("classes.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

# Train loss ve validation loss'u matplotlib ile çizdiriyoruz.
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot.png")

