import cv2
import os
import matplotlib.pyplot as plt
import shutil
import dlib
import numpy as np
from scipy.spatial import distance as dist

classificador_dlib_68_path = 'classificadores\shape_predictor_68_face_landmarks.dat'
classificador_dlib = dlib.shape_predictor(classificador_dlib_68_path)
detector_face = dlib.get_frontal_face_detector()

def anotar_rosto(imagem):
    retangulos = detector_face(imagem, 1)
    if len(retangulos) == 0:
        return imagem

    for k, d in enumerate(retangulos):
        cv2.rectangle(imagem, (d.left(), d.top()), (d.right(), d.bottom()), (255,255,0), 4)
    return imagem


def pontos_marcos_faciais(imagem):
    retangulos = detector_face(imagem, 1)

    if len(retangulos) == 0:
        return None
    marcos = []

    for ret in retangulos:
        marcos.append(np.matrix([[p.x, p.y] for p in classificador_dlib(imagem,ret).parts()]))

    return marcos
    

def anotar_marcos_faciais(imagem, marcos):

    for marco in marcos:
        for idx, ponto in enumerate(marco):
            centro = (ponto[0,0], ponto[0,1])
            cv2.circle(imagem, centro, 3 , (255,255,0), -1)
            cv2.putText(imagem, str(idx), centro, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return imagem


FACE = list(range(17, 68))
FACE_COMPLETA = list(range(0, 68))
LABIO = list(range(48, 61))
SOBRANCELHA_DIREITA = list(range(17, 22))
SOBRANCELHA_ESQUERDA = list(range(22, 27))
OLHO_DIREITO = list(range(36, 42))
OLHO_ESQUERDO = list(range(42, 48))
NARIZ = list(range(27, 35))
MANDIBULA = list(range(0, 17))



def aspeco_razao_olhos(pontos_olhos):
    a = dist.euclidean(pontos_olhos[1], pontos_olhos[5])
    b = dist.euclidean(pontos_olhos[2], pontos_olhos[4])
    C = dist.euclidean(pontos_olhos[0], pontos_olhos[3])

    aspecto_razao = (a + b) / (3.0*C)

    return aspecto_razao

def aspeco_razao_boca(pontos_boca):
    a = dist.euclidean(pontos_boca[3], pontos_boca[9])
    b = dist.euclidean(pontos_boca[2], pontos_boca[10])
    c = dist.euclidean(pontos_boca[4], pontos_boca[8])
    d = dist.euclidean(pontos_boca[0], pontos_boca[6])

    aspecto_razao = (a + b + c) / (3.0*d)

    return aspecto_razao



def anotar_marcos_casca(imagem, marcos, regiao):
    retangulos = detector_face(imagem)

    if len(retangulos) == 0:
        return imagem
    
    for idx, ret in enumerate(retangulos):
        marco = marcos[idx]

        pontos = cv2.convexHull(marco[regiao])
        cv2.drawContours(imagem, [pontos], 0, (0,255,0), 2)
        if regiao is not LABIO:
            pontos = cv2.convexHull(marco[regiao])
            cv2.drawContours(imagem, [pontos], 0, (0,255,0), 2)

        
    return imagem