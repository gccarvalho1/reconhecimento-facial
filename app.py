import cv2
import dlib
import matplotlib.pyplot as plt
import seaborn as sns

caminho_imagem = r'C:\Users\gabri\OneDrive\Documentos\Alura ~ Codes\Deep Learning\RF - Computing\imagens\teste\s01_11.jpg'
###CONVERSÃO RGB
imagem = cv2.imread(caminho_imagem)
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
#####

# Imagem gray pro HAAR identificar (Classificador em cascata)
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

classificador = cv2.CascadeClassifier(r"C:\Users\gabri\OneDrive\Documentos\Alura ~ Codes\Deep Learning\RF - Computing\classificadores\haarcascade_frontalface_default.xml")

faces = classificador.detectMultiScale(imagem_gray, 1.1, 5)
#Fim do classificador
imagem_anotada = imagem.copy()
face_imagem = 0

def bourdingbox(faces, face_imagem):
    for (x, y, w, h) in faces:
        cv2.rectangle(imagem_anotada, (x,y), (x+w, y+h), (0, 0, 255), 2)
        face_imagem+= 1
        imagem_roi = imagem[y:y+h, x:x+w]
        imagem_roi = cv2.cvtColor(imagem_roi, cv2.COLOR_RGB2BGR)
        cv2.imwrite("face_" + str(face_imagem) +".png", imagem_roi)

imagem_analisada = bourdingbox(faces, face_imagem)


plt.figure(figsize=(10,5))
plt.imshow(imagem_anotada)
plt.show()



cv2.waitKey(0)