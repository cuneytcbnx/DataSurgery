import xml.etree.ElementTree as ET
import shutil
import glob
import os

# Verisetimizdeki öznitelikleri tutmak için değişken tanımlıyoruz.
number = None
tirads = None

# Verisetimizi benign ve malign (iyi huylu / kötü huylu) olarak ayırmak için bu kod bloğunu yazıyoruz.
# Bu işlemi gerçekleştirirken aynı zamanda etiketlenmemiş görüntüler veya etiket bulunan ve görüntüsü
# olmayan verileri ayrıştırıyoruz(siliyoruz).
for path in glob.glob("dataset/*.xml"):
    # XML dosyası içindeki ağaçta dolaşmaya başlıyoruz.
    tree = ET.parse(path)
    root = tree.getroot()
    # Xml dosyasındaki number özniteliğini bulup değişkende tutuyoruz
    for boxes in root.iter('number'):
        number = boxes.text
    # Xml dosyasındaki tirads özniteliğini bulup değişkende tutuyoruz
    for boxes in root.iter('tirads'):
        # tirads özniteliği etiketlenmemiş ise kaydetmiyoruz.
        if boxes.text == "":
            print("None")
        # tirads özniteliği etiketlenmiş ise değişkende tutuyoruz.
        else:
            tirads = boxes.text


    for boxes in root.iter('mark'):
        for box in boxes.findall("image"):
                # triads 2 veya 3 değerinde ise görüntüyü Benigns klasörüne taşıyoruz.
                if tirads == "2" or tirads == "3":
                    path = "dataset/" + number + "_" + box.text + ".jpg"
                    print("path:",path)
                    # Etiket dosyasının(xml) görüntüsü olmama durumu olduğu için görüntüyü taşıma
                    # işlemini kontrol ettiriyoruz.
                    try:
                        shutil.move(path,"dataset/Benigns")
                    except Exception as e:
                        print(str(e))
                # triads 4a, 4b, 4c veya 5 değerinde ise görüntüyü Maligns klasörüne taşıyoruz.
                elif tirads == "4a" or tirads == "4b" or tirads == "4c" or tirads == "5":
                    path = "dataset/" + number + "_" + box.text + ".jpg"
                    print("path:", path)
                    try:
                        shutil.move(path, "dataset/Maligns")
                    except Exception as e:
                        print(str(e))



