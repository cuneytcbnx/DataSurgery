# Gerekli kütüphaneleri içe aktarıyoruz.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow import keras

# Eğitim yapmak için oluşturacağımız CNN ağı sınıfını oluşturuyoruz.
class CreateCNN:
    # Kurucu fonksiyonuna 4 adet parametre girdisi tanımlıyoruz.
    # width: Görüntü genişliği
    # height: Görüntü yüksekliği
    # depth: Görüntü derinliği (Kanal sayısı)
    # classes: Sınıf sayısı

    def build(width, height, depth, classes):

        model = Sequential()
        inputShape = (height, width, depth)

        chanDim = -1

        #  CONV => RELU => POOL
        #POOL katmanımız, uzamsal boyutları 430 x 360'dan 32 x 32'ye hızlı bir şekilde azaltmak için 3 x 3 Pool boyutu kullanır.
        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        #Birden fazla CONV ve RELU katmanını birlikte istiflemek (hacmin uzamsal boyutlarını azaltmadan önce), daha zengin bir özellik kümesi öğrenmemizi sağlar.
        #Filtre boyutumuzu 32'den 64'e çıkarıyoruz. Ağda ne kadar derine inersek, hacmimizin uzamsal boyutları o kadar küçülür ve o kadar çok filtre öğreniriz.
        #Uzamsal boyutlarımızı çok hızlı küçültmemek için maksimum havuz boyutunu 3 x 3'ten 2 x 2'ye düşürdük.
        #Dropout yine bu aşamada gerçekleştirilir.
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        #Burada filtre boyutumuzu 128'e çıkardık. Fazla overfitting'i(ezberleme) tekrar azaltmak için düğümlerin %25'ini bırakıyoruz.
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU katmanlarının ilk (ve tek) kümesi
        #Tam bağlı katman, düzeltilmiş doğrusal birim aktivasyonu ve toplu normalleştirme ile Dense(1024) tarafından belirtilir.
        #Bırakma(Dropout) son bir kez gerçekleştirilir - bu sefer eğitim sırasında düğümlerin %50'sini bırakıyoruz.
        #Modeli, her sınıf etiketi için tahmin edilen olasılıkları döndürecek bir softmax sınıflandırıcı ile tamamlıyoruz.
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax sınıflandırıcı
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # Yapılandırılmış ağ mimarisini(Model) geriye döndürüyoruz.
        return model