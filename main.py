import cv2
from librerias import filtros, binarizacion

def main():

    #comencemos por cargar la imagen en cuestion
    imagen = cv2.imread("./src/Jit2.JPG")

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

    return 0

if __name__ == "__main__":
    main()