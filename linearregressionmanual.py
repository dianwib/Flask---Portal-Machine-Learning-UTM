import numpy as np
from sklearn.datasets import load_diabetes


class ManualLinearRegression:
    def __init__(self):
        return

    # Menambahkan angka 1 pada matrix pada setiap awal kolom
    def build_matrix(self, x):
        # print(x.shape)
        [num_of_data, dim] = x.shape
        new_x = np.ones((num_of_data, dim + 1))
        new_x[:, 1:] = x
        return new_x

    # Membuat sebuah model
    def training(self, data_training, target_training, num_of_iteration, alfa):
        [num_of_data, dim] = data_training.shape
        weight = np.zeros((num_of_iteration, dim + 1))  # +1 for bias
        temp_weight = np.random.rand(1, dim + 1)
        temp_data = self.build_matrix(data_training)
        # print(temp_data)
        for i in range(num_of_iteration):
            for j in range(num_of_data):
                prediction = temp_weight.dot(temp_data[j])
                error = prediction - target_training[j]
                temp_weight = temp_weight - alfa * error * temp_data[j]

                # if i == 4 and j == 0:
                #     print("bobot =", temp_weight)
            weight[i] = temp_weight
        return weight

    # Generalisasi / testing data test dengan bobot / weight
    def testing(self, data_test, weight):
        num_of_data, feature = data_test.shape
        num_of_weight, dim = weight.shape
        # print(num_of_data, feature, dim)
        data_predict = np.zeros(num_of_data)
        for i in range(num_of_data):
            predict = weight[num_of_weight - 1][0]

            for j in range(1, dim):
                predict += weight[num_of_weight - 1][j] * data_test[i][j - 1]
            data_predict[i] = predict
            # if i == 25:
            #     print("data 25 =", data_test[i], data_predict[i])
        return data_predict

    # Menghitung akurasi dengan r-square
    def accurate(self, target_test, predict):
        num_of_target = target_test.shape[0]
        sstot = 0
        ssrest = 0
        for i in range(num_of_target):
            sstot += (target_test[i] - np.mean(target_test)) ** 2
            ssrest += (target_test[i] - predict[i]) ** 2
        return 1 - ssrest / sstot

if __name__ == "__main__":
    # Penyiapan dataset
    data, target = load_diabetes(return_X_y=True)
    data_training, target_training = data[:400, :], target[:400]
    data_test, target_test = data[400:, :], target[400:]

    # Deklarasi object
    stochasticGD = ManualLinearRegression()
    w = stochasticGD.training(data_training, target_training, 350, 0.02)
    model = str(w[350 - 1][0])
    print(model)
    for index_weight in range(1, 11):
        print(str(w[350 - 1][index_weight]))
        model = model + (' + ' + str(w[350 - 1][index_weight]) + (' x%s'%(index_weight+1)))
        print(model)
    predict = stochasticGD.testing(data_training, w)
    print("data 25", data_training[24], predict, target_training[24])
    acc = stochasticGD.accurate(target_test, predict)
    print(target_test.shape)
    print("Akurasi dari Linear Regression menggunakan Stochastic Gradient Descent =", acc * 100, "%")
