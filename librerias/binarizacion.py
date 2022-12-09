import cv2
import numpy as np

#umbral = int(input('valor del Umbral: '))
#img = cv2.imread('lenna_main.jpg', 0)
#cv2.imshow("imagen original", img)
#largo, alto = img.shape

def obtenerHistograma(imagenBN):
    '''retorna el histograma de una imagen en una lista de 0 a 255, es decir, el histograma de todos los niveles de gris de la imagen'''
    contador = 0
    histograma = []

    largo, alto = imagenBN.shape

    for i in range(256):
        contador = 0

        for j in range(largo):
            for k in range(alto):
                
                if imagenBN[j,k] == i:
                    contador += 1
        
        histograma.append(contador)

    return histograma


def umbralar(img, largo, alto, umbral):
    img2 = np.zeros(img.shape, img.dtype)

    for x in range(largo):
        for y in range(alto):
            if(img[x, y]<umbral):
                img2[x, y] = 255
            else:
                img2[x,y] = 0


    #cv2.imshow("imagen umbralada", img2)


    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img2

def VarianzaOtzu(histograma, iteracion):
    totalPixeles = 0
    parcialPixeles = 0
    pesoB = 0
    pesoF = 0
    mediaB = 0
    mediaF = 0
    varianzaB = 0
    varianzaF = 0
    varianzaClase = 0

    #obtenemos el total de pixeles
    for i in range(256):
        totalPixeles += histograma[i]
    
    #obtenemos el total de pixeles que estaran por debajo del umbral
    for i in range(iteracion):
        parcialPixeles += histograma[i]

    #obtenemos el peso y la media del fondo
    for i in range(iteracion):
        pesoB += float(histograma[i] / totalPixeles)

        if parcialPixeles == 0:
            mediaB = 0
        else:
            mediaB += float((i*histograma[i]/parcialPixeles))

    #obtenemos la varianza el fondo
    for i in range(iteracion):
        if parcialPixeles == 0:
            varianzaF = 0
        else:
            varianzaB += float(((i-mediaB)**2 * histograma[i]) / parcialPixeles)
    
    #obtenemos el peso y la media del frente

    for i in range(iteracion, 256):
        pesoF += float(histograma[i] / totalPixeles)

        if parcialPixeles == totalPixeles:
            mediaF = 0
        else:
            mediaF += float((i * histograma[i]) / (totalPixeles - parcialPixeles))
    
    #obtenemos la varianza del frente

    for i in range(iteracion, 256):

        if parcialPixeles  == totalPixeles:
            varianzaF = 0
        else:
            varianzaF += float(((i-mediaF)**2 * histograma[i]) / (totalPixeles - parcialPixeles))

    #obtenemos la varianza dentro de la clase

    varianzaClase = float((pesoB * varianzaB) + (pesoF * varianzaF))

    print("varianza de clase: {}\nvarianza fondo: {}\nvarianza frente: {}".format(varianzaClase, varianzaB, varianzaF))

    return varianzaClase

def OTSU(histograma):
    mejorUmbral = 0
    umbralActual = 0
    varianzaActual = 0
    mejorVarianza = 0

    for i in range(256):

        if i == 0:
            mejorUmbral = i
            mejorVarianza = VarianzaOtzu(histograma, i)
        else:
            umbralActual = i
            varianzaActual = VarianzaOtzu(histograma, i)

            if mejorVarianza < varianzaActual:
                mejorUmbral = mejorUmbral
            else:
                mejorUmbral = umbralActual
                mejorVarianza = varianzaActual
        
    return mejorUmbral