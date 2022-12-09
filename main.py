import cv2
from librerias import filtros, binarizacion
import numpy as np
import math

def main():

    #comencemos por cargar la imagen en cuestion
    imagen = cv2.imread("./src/Jit1.JPG")

    print("sacando escala de grises...")
    #sacamos la escala de grises de la imagen
    imagenBN = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    
    print("suavizando...")
    #apliquemos suavizado para posteriormente umbralar
    kernelGauss = filtros.kernelGauss(5, 1)
    imagenBNExpandida = filtros.expandirImagen(imagenBN, kernelGauss)
    imagenBNSuavizada = filtros.filtrarImagen(imagenBNExpandida, imagenBN, kernelGauss)

    
    print("umbralando...")
    #ahora umbralamos con OTSU
    histograma = binarizacion.obtenerHistograma(imagenBNSuavizada)
    umbral = binarizacion.OTSU(histograma)
    imagenUmbralada = binarizacion.umbralar(imagenBNSuavizada, imagen.shape[0], imagen.shape[1], umbral)

    
    #imprimimos las imagenes para ver como ha resultado todo hasta el momento

    cv2.imshow("original", imagen)
    cv2.imshow("umbralada", imagenUmbralada)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #aplicamos morfologia para cerrar grietas obtenidas en el umbral de OTSU
    estructura = np.ones((round(imagen.shape[0]*0.065*1.7),round(imagen.shape[0]*0.065)), np.uint8)
    imagenMorph = cv2.morphologyEx(imagenUmbralada, cv2.MORPH_OPEN, estructura)

    #visualizamos

    #cv2.imwrite("./umbralada.jpg", imagenUmbralada)
    #cv2.imwrite("./clausuradda.jpg", imagenMorph)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #encontramos los contornos de la imagen ya procesada

    contornos, jerarquia = cv2.findContours(imagenMorph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #al menos en este experimento, a base de ejecuciones, determinamos manualmente que los centroides de los tomates que buscamos, son el 2 y el 0
    #así que ejecutamos el algoritmo para obtener esos centroides y poder trabajar la parte restante

    #creemos tambien una lista que me guarde los centroides de interes

    centroides_interes = []
    interes = [0,2]

    for i in interes:
        centroide = contornos[i]
        momentos = cv2.moments(centroide)

        coordX = int(momentos["m10"]/momentos["m00"])
        coordY = int(momentos["m01"]/momentos["m00"])

        #guardamos en el arreglo el centroide
        centroides_interes.append([coordX, coordY])

        #dibujamos la ubicacion de los centoides

        #cv2.circle(imagenMorph, (coordX, coordY), 5, (0,255,0), -1)  <- Esta linea dibuja el centroide sobre la imagen, para comprobar donde esta el centroide descomente esta linea

    #cisualizamos los centorides

    #cv2.imshow("centroides", imagenMorph)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #comenzamos a trabajar con el centrooide del tomate 2 <- ubicado en el centroide  2

    #como este tomate se encuentra alineado al eje horizontal, este es el caso más facil

    contadorPixeles = 0
    pixelCentral = centroides_interes[1] #<- recordemos que primero almacenamos el centroide 4 y luego el 2, por eso el indice 1
    pixel = pixelCentral
    bandera = True
    direccion = True
    tomate = contornos[0]
    pixel_inferior = 0
    pixel_superior = 0

    mayor_distancia = 0
    pixelHonorario = []

    print(pixel)
    print(imagenMorph[160, 375])

    while(bandera or direccion):
        
        if imagenMorph[pixel[1], pixel[0]] > 0:
            contadorPixeles+=1
            #imagenMorph[pixel[1], pixel[0]] = 0 # <- Esto pinta la imagen ya umbralada y con morfologia aplicada, así que conviene solo usarla para probar
            bandera = True

            if(direccion):
                pixel[0] -=1 #<- movemos horizontalmente el pixel analizado, ya que la distancia que queremos encontrar es está en el eje de las absisas
            else:
                pixel[0] +=1
        else:
            if(direccion == True):
                direccion = False
                pixel_inferior = pixel
                pixel = [pixelCentral[0]+1, pixelCentral[1]]
            else:
                bandera = False
                pixel_superior=pixel
        

    #para comprobar tracemos una linea entre el pixel con la mayor distancia del centro
    cv2.line(imagen, pixel_inferior, pixel_superior, (0,255,0), 5)

    #imprimimos el contador y la imagen resultante para comprobar resultados
    print("el pixel inferior de la medicion para el tomate 2: {} y el duperior: {}".format(pixel_inferior, pixel_superior))
    print("el tomate 2 mide de punta a punta {} pixeles".format(contadorPixeles))
    #cv2.imshow("centroides", imagen)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #aplicamos el proceso por segunda vez, pero esta vez haciendola cuenta del centro hasta el borde derecho

    '''pixel = pixelCentral
    bandera = True

    while(bandera):
        
        if imagenMorph[pixel[0], pixel[1]] > 0:
            contadorPixeles+=1
            bandera = True
            pixel[0] +=1 #<- movemos horizontalmente el pixel analizado, ya que la distancia que queremos encontrar es está en el eje de las absisas
        else:
            bandera = False'''

    #Ahora aplicamos el procesamiento al segundo tomate, ubicado en el centroide 0

    contadorPixeles = 0
    pixelCentral = centroides_interes[0]
    pixel = pixelCentral
    tomate = contornos[0] #<- corresponde al numero de controno [0,4] que fueron los que encontramos al preprocesar la imagen
    #tenemos que encontrar uno de los pixeles más alejados del pixel central, y para eso, sacando uno de los dos, ya tenemos la panediente, con la cual podemos sacar el otro, y dibujar la linea

    #primero usamos un bucle para que el pixelse aleje a la izquierda del pixel central tanto como sea posible

    mayor_distancia = 0
    pixelHonorario = []

    for pixel in tomate:
        #print(pixel[0])
        distancia = np.sqrt((pixelCentral[0]-pixel[0][0])**2 + (pixelCentral[1]-pixel[0][1])**2)

        if mayor_distancia == 0:
            mayor_distancia = distancia
            pixelHonorario = pixel[0]
        else:
            if(distancia>mayor_distancia):
                mayor_distancia = distancia
                pixelHonorario = pixel[0]
            else:
                mayor_distancia = mayor_distancia
                pixelHonorario = pixelHonorario

    #para comprobar tracemos una linea entre el pixel con la mayor distancia del centro
    cv2.line(imagen, pixelHonorario, pixelCentral, (0,255,0), 5)

    #ahora que tenemos la mitad de la distancia mas larga, el dobe de esa distancia, sera la distancia en pixeles de punta a punta del tomate
    distanciaTotal = 2*mayor_distancia

    #finalmente, para terminar el trazado de la linea, al pixel que localizamos, le sumamos el doble de su suma conel pixel central
    segundoExtremo = [pixelCentral[0]+ (pixelCentral[0]-pixelHonorario[0]), pixelCentral[1] + (pixelCentral[1]-pixelHonorario[1])]

    #imprimamos su linea, a ver que ocurre
    cv2.line(imagen , pixelCentral, segundoExtremo, (0,255,0), 5)

    #listo, aparentemente ya esta todo, habra que haerle algunos ajustes con la imagenn completa
    print("el pixel inferior de la medicion para el tomate 4: {} y el duperior: {}".format(pixelHonorario, segundoExtremo))
    print("La distancia del tomate 4 de extremo a extremo en pixeles es de {} pixeles".format(math.floor(distanciaTotal)))
    #imprimimos la imagen resultante

    #cv2.imwrite("./final.jpg", imagen)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #print(tomate[0])

    #guardamos las imagenes del proceso que hicimos
    cv2.imwrite("./evidencias/1-EscalaGrises.jpg",imagenBN)
    cv2.imwrite("./evidencias/2-Expandida.jpg",imagenBNExpandida)
    cv2.imwrite("./evidencias/3-Suavizada.jpg",imagenBNSuavizada)
    cv2.imwrite("./evidencias/4-SegmentadaConOTSU.jpg",imagenUmbralada)
    cv2.imwrite("./evidencias/5-Mofologia Intermedia.jpg",imagenMorph)
    cv2.imwrite("./evidencias/6-ResultadoFinal.jpg",imagen)

    return 0

    

if __name__ == "__main__":
    main()