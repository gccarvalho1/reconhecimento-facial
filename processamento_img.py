import cv2
import os
import matplotlib.pyplot as plt
import dlib
import numpy as np
from classificador import *
from io import BytesIO
from  IPython.display import clear_output, Image, display
from PIL import Image
from PIL import ImageTk
import time
import tkinter as tk

root = tk.Tk()
root.title("Video")

# Criar um rótulo para exibir o vídeo
video_label = tk.Label(root)
video_label.pack()


# Definindo caminhos
faces_caminho = './imagens/cropped_faces/'
faces_path_treino = r'./imagens/treino/'
faces_path_teste = './imagens/teste/'

# Criar diretórios se não existirem
os.makedirs(faces_path_treino, exist_ok=True)
os.makedirs(faces_path_teste, exist_ok=True)

modelo_eingenfaces = cv2.face.LBPHFaceRecognizer_create()


# Listar imagens de treino e teste
lista_faces_treino = [f for f in os.listdir(faces_path_treino) if os.path.isfile(os.path.join(faces_path_treino, f))]
lista_faces_treino.sort()  # Ordenando a lista de arquivos de treino

lista_faces_teste = [f for f in os.listdir(faces_path_teste) if os.path.isfile(os.path.join(faces_path_teste, f))]
lista_faces_teste.sort()  # Ordenando a lista de arquivos de teste

dados_treinamento, sujeitos = [], []

for j, arq in enumerate(lista_faces_treino):
    k = cv2.imread(faces_path_treino + lista_faces_treino[j], cv2.IMREAD_GRAYSCALE)
    k = cv2.resize(k, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    k = k.copy()
    dados_treinamento.append(k)
    sujeito = int(arq[1:3])
    sujeitos.append(sujeito)

train = modelo_eingenfaces.train(dados_treinamento, np.asarray(sujeitos, dtype=np.int32))

sujeitos_treino = []
## ------------ METODO DE ACCURÁCIA DO MODELO ----------#
total = len(lista_faces_teste)
def test_model():
    acertos = 0
    for k, arq2 in enumerate(lista_faces_teste):
        imagem = cv2.imread(faces_path_teste + lista_faces_teste[k], cv2.IMREAD_GRAYSCALE)
        imagem = cv2.resize(imagem, (200,200), interpolation=cv2.INTER_LANCZOS4)
        sujeito = int(arq2[1:3])
        sujeitos_treino.append(sujeito)
        valor, predict = modelo_eingenfaces.predict(imagem)
        if valor == sujeitos_treino[k]:
            acertos+=1
    return acertos

# ------------- FIM DO METODO DE ACURÁCIA-----------#

classificador_dlib_68_path = 'classificadores\shape_predictor_68_face_landmarks.dat'
classificador_dlib = dlib.shape_predictor(classificador_dlib_68_path)
detector_face = dlib.get_frontal_face_detector()


def testar_individual(x,y):
    imagem = cv2.imread(faces_path_teste + f's{x}_{y}.jpg', cv2.IMREAD_GRAYSCALE)
    imagem = cv2.resize(imagem, (200,200), interpolation=cv2.INTER_LANCZOS4)
    

    #------------Analise de teste----------#
    valor, predict = modelo_eingenfaces.predict(imagem)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title('Sujeito preditado:' + str(x))
    plt.imshow(imagem, cmap='grey')
    plt.subplot(122)
    plt.title('Sujeito acertado:' + str(x))
    imagem_predict = cv2.imread(faces_path_treino + f's{valor}_01.jpg', cv2.IMREAD_GRAYSCALE)
    imagem_predict = cv2.resize(imagem_predict, (200,200), interpolation=cv2.INTER_LANCZOS4)
    plt.imshow(imagem_predict, cmap='grey')
    plt.show()

    #-------- FIM DO MODELO DE TESTES-----#


def desenhar_face(imagem):

    marcos_faciais = pontos_marcos_faciais(imagem)
    imagem_anotada = anotar_rosto(imagem)
    #imagem_anotada = anotar_marcos_faciais(imagem_anotada, marcos_faciais)
    if marcos_faciais is not None:
        #imagem_anotada = aspeco_razao_boca(marcos_faciais[0][LABIO])
        #imagem_anotada = round(imagem_anotada, 3)
        imagem_anotada = anotar_marcos_casca(imagem_anotada, marcos_faciais, LABIO)
    return imagem_anotada

def padronizar_imagem(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (500, 400))
    return frame

def exibir_video(frame):
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.update_idletasks()

def capturar_video():
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    while True:
        captura_ok, frame = video.read()
        if not captura_ok:
            print("Erro: Não foi possível ler o frame da webcam.")
            break
        frame = desenhar_face(frame)
        frame = padronizar_imagem(frame)
        exibir_video(frame)

        # Atualizar a janela tkinter
        root.update()

    video.release()

# Iniciar captura de vídeo
capturar_video()

# Iniciar o loop principal do tkinter
root.mainloop()
#desenhar_face()

#testar_individual(25, 11)
#true = test_model() # testa tudo
#print(f"Taxa de acerto: {true / total * 100:.2f}%")
