import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self):
        return
    
    def predict(self, data, tempCenter):
        dataLabel = np.zeros([len(data)], dtype=int)
        for i in range(len(data)):
            eDist = np.zeros(len(tempCenter))
            #print(data[i])
            #--
            #-- menghitung jarak data ke-i ke tiap center
            #--
            for j in range(len(eDist)):
                eDist[j] = 0
                for k in range(len(data[i])):
                    eDist[j] = eDist[j] + ( data[i,k] - tempCenter[j,k] ) ** 2
                eDist[j] = np.sqrt(eDist[j])

            #--
            #-- mencari jarak terpendek
            #--   
            minEDist = 0
            for l in range(len(eDist)):
                if(eDist[minEDist] > eDist[l]):
                    minEDist = l
            dataLabel[i] = minEDist
        return dataLabel


    def training(self, data, center, itteration):
        #print('center :\n',center)
        tempCenter = center;
        for loop in range(itteration):
            #--
            #-- memprediksi semua data menggunakan tempCenter
            #--
            dataLabel = self.predict(data, tempCenter)

            #--
            #-- menyimpan [jumlah nilai attribut semua data, dan banyaknya data] di tiap class
            #--
            centerMean = np.zeros([len(tempCenter), len(tempCenter[0])+1])
            for i in range(len(data)):
                for j in range(len(data[i])):
                    centerMean[dataLabel[i],j] += data[i,j]
                centerMean[dataLabel[i], len(centerMean[0])-1] += 1
            #print(centerMean)

            #--
            #-- menghitung mean dari centerMean
            #--
            newCenter = np.zeros([len(tempCenter), len(tempCenter[0])])
            for i in range(len(newCenter)):
                for j in range(len(newCenter[i])):
                    newCenter[i,j] = centerMean[i,j] / centerMean[i, len(centerMean[0])-1]
                    #print(centerMean[i, len(centerMean[0])-1])
            #print('new Center',loop,':\n',newCenter)
            tempCenter = newCenter
            
        return tempCenter
    
    def silhouette(self, indexPi, data, dataLabel):
        #-- 
        #-- menyimpan [jumlah jarak indexPi ke semua data, dan banyaknya data] di tiap class
        #-- 
        tempMeanDist = np.zeros([len(center), 2]) 
        for i in range(len(dataLabel)):
            if i != indexPi:
                eDist = 0
                for j in range(len(data[i])):
                    eDist += (data[indexPi,j] - data[i,j]) ** 2
                eDist = eDist ** (1/2)
                tempMeanDist[dataLabel[i],0] += eDist
                tempMeanDist[dataLabel[i],1] += 1

        #--
        #-- menghitung rata-rata dari tempMeanDist
        #--      
        meanDist = np.zeros(len(tempMeanDist))
        for i in range(len(meanDist)):
            meanDist[i] = tempMeanDist[i,0] / tempMeanDist[i,1]

        #--
        #-- class dari data ke-indexPi
        #-- untuk menentukan nilai a/silhouetteA
        #-- 
        silhouetteClass = dataLabel[indexPi]
        silhouetteA = np.array(meanDist[silhouetteClass])
        silhouetteB = np.delete(meanDist, silhouetteClass)

        #--
        #-- untuk 2 b/silhouetteB lebih, memilih yang terkecil
        #-- 
        silhouetteBminInd = 0
        for i in range(len(silhouetteB)):
            if silhouetteB[i] < silhouetteB[silhouetteBminInd]:
                silhouetteBminInd = i
        silhouetteB = silhouetteB[silhouetteBminInd]

        silhouette = ( silhouetteB - silhouetteA ) / np.maximum( silhouetteA, silhouetteB )
        return silhouette

    def avgSilhouette(self, data, dataLabel):
        sumSilhouette = 0
        numSilhouette = 0
        for i in range(len(data)):
            sumSilhouette = sumSilhouette + self.silhouette(i, data, dataLabel)
            numSilhouette = numSilhouette + 1
        return sumSilhouette / numSilhouette

    def findCenter(self, data, numCenter):
        minValue = np.amin(data, axis=0)
        maxValue = np.amax(data, axis=0)
        rangeValue = maxValue - minValue
        center = []
        for i in range(numCenter):
            center.append(rangeValue/(numCenter-1)*i)
        return np.array(center)

    def findRandomCenter(self, data, numCenter):
        minValue = np.amin(data, axis=0)
        maxValue = np.amax(data, axis=0)
        rangeValue = maxValue - minValue
        center = []
        for i in range(numCenter):
            tempCenter = []
            for j in range(len(rangeValue)):
                tempCenter.append(np.random.randint(rangeValue[j]) + minValue[j])
            center.append(tempCenter)
        return np.array(center)

    def predict2img(self, data, center):
        result = np.zeros([len(data), len(center)])
        for i in range(len(center)):
            result[data == i] = center[i]
        return result

    def predictEDist(self, data, tempCenter):
        dataLabel = np.zeros([len(data)], dtype=int)
        dataEDist = np.zeros([len(data)])
        for i in range(len(data)):
            eDist = np.zeros(len(tempCenter))
            #print(data[i])
            #--
            #-- menghitung jarak data ke-i ke tiap center
            #--
            for j in range(len(eDist)):
                eDist[j] = 0
                for k in range(len(data[i])):
                    eDist[j] = eDist[j] + ( data[i,k] - tempCenter[j,k] ) ** 2
                eDist[j] = np.sqrt(eDist[j])

            #--
            #-- mencari jarak terpendek
            #--   
            minEDist = 0
            for l in range(len(eDist)):
                if(eDist[minEDist] > eDist[l]):
                    minEDist = l
            dataLabel[i] = minEDist
            dataEDist[i] = eDist[minEDist]
        return dataLabel, dataEDist
    
def load_images_from_folder(folder):
    images = []
    fileNames = []
    for filename in os.listdir(folder):
        if(os.path.splitext(filename)[1] == '.jpg'):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                fileNames.append(filename)
    return images

if __name__ == "__main__":
    folderName = "static/assets/images/msra-b-0/dataset"
    images = load_images_from_folder(folderName)

    data = images[0]
    data = cv2.resize(data, (256, 256))
    #cv2.imshow('image data', data)
    dataDim = data.shape
    data = data.reshape(dataDim[0]*dataDim[1],dataDim[2])

    # Deklarasi object
    kmeans = Kmeans()
    #center = kmeans.findCenter(data, 3)
    center = kmeans.findRandomCenter(data, 3)
    center = kmeans.training(data, center, 1)
    print('center :\n',center)
    predict = kmeans.predict(data, center)
    predict = kmeans.predict2img(predict, center)
    predict = predict.reshape(dataDim)
    predict = np.array(predict, dtype=np.uint8)
    cv2.imshow('image data', predict)

