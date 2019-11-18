import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


class ManualLogisticRegression:
    def __init__(self):
        return

    def factorial(self, itter):
        hasil = 1
        while itter >= 1:
            hasil = hasil * itter
            itter = itter - 1
        return hasil

    def e(self):
        x = 0
        for i in range(20):
            x = x + 1/self.factorial(i)
        return x

    def buildMatrix(self, x):
        [numOfData, dim] = x.shape
        newX=np.ones((numOfData, dim+1))
        newX[:, 1:] = x
        return newX

    def training(self, data_training, target_training, numOfIteration, alfa):
        [numOfData, dim] = data_training.shape
        weight = np.zeros((numOfIteration, dim+1))
        tempWeight = np.random.randn(1, dim+1)
        tempWeight=np.zeros([1, dim+1])
        tempData=self.buildMatrix(data_training)
        
        for i in range(numOfIteration):        
            for j in range(numOfData):
                prediction = self.predict(tempData[j], tempWeight)
                tempWeight = tempWeight + alfa * (target_training[j] - prediction) * prediction *\
                             (1 - prediction) * tempData[j]
                #print(tempWeight)
            weight[i]=tempWeight
        
        return weight

    def predict(self, x, w):
        return 1 / (1 + self.e() ** (- (w.dot(x))))

    def testing(self, data_test, weight):
        data_test = self.buildMatrix(data_test)
        predictionMatrix = np.zeros(len(data_test))

        for i in range(len(data_test)):
            predictValue = self.predict(data_test[i], weight)
            if predictValue < 0.5:
                predictionMatrix[i] = 0
            else:
                predictionMatrix[i] = 1
        return predictionMatrix

    def accurate(self, target_test, predict):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(target_test)):
            if target_test[i] == 1 and predict[i] == 1:
                TP = TP + 1
            if target_test[i] == 1 and predict[i] == 0:
                FP = FP + 1
            if target_test[i] == 0 and predict[i] == 0:
                TN = TN + 1
            if target_test[i] == 0 and predict[i] == 1:
                FN = FN + 1

        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        return prec, recall

if __name__ == "__main__":
    # Penyiapan dataset
    data, target = load_breast_cancer(return_X_y=True)
    data_training, data_test, target_training, target_test = train_test_split(data, target, test_size=0.30, random_state=0)
    print(data_test.shape)
    data_training = normalize(data_training)
    data_test = normalize(data_test)

    # Deklarasi objects
    logisticRegression = ManualLogisticRegression()
    w = logisticRegression.training(data_training, target_training, 50, 0.3)
    predict = logisticRegression.testing(data_test, w[len(w)-1])
    precision, recall = logisticRegression.accurate(target_test, predict) # [precision, recall]
    print("Akurasi menggunakan incremental learning [precision, recall] =",
        precision * 100, ",", recall * 100, "%")
    #output untuk data ke 320 - 330 kemudian keluarkan confusion matrix nya
    print(data_test[4], target_test[4], predict[4])
