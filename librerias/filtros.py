import cv2
import numpy as np

def kernelGauss(dimensionFiltro, sigma):
    
    limiteFiltro = int((dimensionFiltro - 1)/2)
    kernel = np.zeros((dimensionFiltro, dimensionFiltro), float)

    i = 0
    for x in range(-limiteFiltro, limiteFiltro+1):
        j = 0
        for y in range(-limiteFiltro, limiteFiltro+1):
            kernel[i,j] = np.exp(-((pow(x, 2) + pow(y, 2)) / (2 * pow(sigma, 2)))) / (2 * np.pi * sigma**2)
            
            j+=1
        i+=1

    print(np.matrix(kernel))
    
    return kernel

def kernelLoG(dimensionFiltro, sigma):

    limiteFiltro = int((dimensionFiltro - 1)/2)
    kernel = np.zeros((dimensionFiltro, dimensionFiltro), float)

    i = 0
    for x in range(-limiteFiltro, limiteFiltro+1):
        j = 0
        for y in range(-limiteFiltro, limiteFiltro+1):
            kernel[i,j] = (1 / (2 * np.pi * sigma**4)) * (2-((x**2 + y**2)/sigma**2)) * np.exp(-((x**2 + y**2) / (2 * sigma**2))) 
            
            j+=1
        i+=1

    print(np.matrix(kernel))
    
    return kernel

def filtrarImagen(imagenBNExpandida, imagenBN, filtro):
    largo, alto = imagenBN.shape
    largoFiltro, altoFiltro = filtro.shape
    filtrada = np.zeros(imagenBN.shape, imagenBN.dtype)

    convolucion = 0
    x = 0
    y = 0

    for i in range(largo):
        for j in range(alto):

            x = i
            for k in range(largoFiltro):
                y = j
                for l in range(altoFiltro):
                    convolucion += float(imagenBNExpandida[x,y]) * filtro[k,l]
                    y += 1
            
                x +=1

            filtrada[i,j] = int(convolucion)
            convolucion = 0
    
    #print(filtrada[:50,:50])

    cv2.imshow("original", imagenBN)
    cv2.imshow("filtrada", filtrada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtrada

def filtrarImagenLoG(imagenBNExpandida, imagenBN, filtro):
    largo, alto = imagenBN.shape
    largoFiltro, altoFiltro = filtro.shape
    filtrada = np.zeros(imagenBN.shape, dtype=float)

    convolucion = 0
    x = 0
    y = 0

    for i in range(largo):
        for j in range(alto):

            x = i
            for k in range(largoFiltro):
                y = j
                for l in range(altoFiltro):
                    convolucion += float(imagenBNExpandida[x,y]) * filtro[k,l]
                    y += 1
            
                x +=1

            filtrada[i,j] = int(convolucion)
            convolucion = 0
    
    #print(filtrada[:50,:50])

    return filtrada

def expandirImagen(imagenBN, filtro):
    largo, alto =  imagenBN.shape
    largoFiltro, altoFiltro = filtro.shape

    expansionFilas = alto + altoFiltro - 1
    expansionColumnas = largo + largoFiltro - 1
    exceso = int((largoFiltro-1)/2)
    print(exceso)

    expandida = np.zeros((expansionColumnas, expansionFilas), imagenBN.dtype)

    for i in range(alto):
        for j in range(largo):
            #if(i> altoFiltro -1 and j> largoFiltro -1 and i<=alto and j <= largo):
            expandida[j+exceso][i+exceso] = imagenBN[j][i]

    #cv2.imshow("expandida", expandida)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return expandida


'''def main():
    dimensionFiltro = 5
    sigma = 1

    kernel = kernelLoG(dimensionFiltro, sigma)

    imagen = cv2.imread("../../src/lenna_main.jpg", 0)
    imagenExpandida = expandirImagen(imagen, kernel)
    filtrarImagen(imagenExpandida, imagen, kernel)


if __name__ == "__main__":
    main()'''