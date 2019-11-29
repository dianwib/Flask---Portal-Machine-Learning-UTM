from flask import Flask, render_template, request
import cv2
from PIL import Image
import linearregressionmanual, logisticregressionmanual
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
@app.route('/')
@app.route('/index')
def hello_world():
	return render_template('index.html')

@app.route('/greyscale_citra',methods=['GET', 'POST'])
def convert_image():
    path_hasil = 'static/assets/greyscale_citra/hasil/'
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = 'static/assets/greyscale_citra/upload/'+ file.filename
        result_name = file.filename[:-4]
        img.save(uploaded_img_path)
        img_input = cv2.imread(uploaded_img_path)
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path_hasil + result_name +'.jpg', gray)
        return render_template('greyscale_citra.html',query_path=uploaded_img_path,img=result_name)
    else:
        return render_template('greyscale_citra.html')

def plotting_model(weight, X, Y):
    plt.title('Prediksi Nilai Diabetes')
    plt.xlabel('Indeks Massa Tubuh')
    plt.ylabel('Nilai Diabetes')
    plt.plot(X, Y, 'k.')
    plt.grid(True)
    data = np.linspace(-0.2, 0.2, 10)
    for i in range(weight.shape[0]):
        y = weight[i, 0] + weight[i, 2] * data
        #print(y)
        plt.plot(data,y,'r-')
    y = weight[weight.shape[0] - 1, 0] + weight[weight.shape[0] - 1, 2] * data
    plt.plot(data,y,'b-')
    plt.savefig('static/assets/img/PlotLinReg.png')

@app.route('/task_linear_regression',methods=['GET', 'POST'])
def task_linear_regression_route():
    if request.method == 'POST':
        # Menyimpan nilai inputan dari web
        weight = 0
        mode = request.form['mode']
        # Training
        if mode == 'Training':
            epochs = int(request.form['query_epochs'])
            print(request.form.get('komposisi', None))
            test_size = (100 - int(request.form.get('komposisi'))) / 100
            option_show_training = request.form['query_tampil_training']
            data, target = load_diabetes(return_X_y=True)
            data_training, data_test, target_training, target_test = train_test_split(data, target, 
                test_size=test_size, random_state=0)
            stochasticGD = linearregressionmanual.ManualLinearRegression()
            weight = stochasticGD.training(data_training, target_training, epochs, 0.05)
            np.save('static/assets/model/model_linreg.npy', weight)
            weight = np.around(weight, decimals=2)
            num_of_weight, dim = weight.shape
            model = 'y = ' + str(round(weight[num_of_weight - 1][0], 2))
            for index_weight in range(1, dim):
                w = round(weight[num_of_weight - 1][index_weight], 2)
                model = model + (' + ' + str(w) + (' * x%s'%(index_weight)))
            # plotting_model(weight, data_training[:, 2], target_training)
            if option_show_training == 'Tidak':
                return render_template('task_linear_regression.html', model=model)
            elif option_show_training == 'Ya':
                return render_template('task_linear_regression.html', model=model, step_trains=weight, epochs=epochs)
        else:
            age = float(request.form['age_input'])
            sex = float(request.form['sex_input'])
            bmi = float(request.form['bmi_input'])
            bp = float(request.form['bp_input'])
            s1 = float(request.form['s1_input'])
            s2 = float(request.form['s2_input'])
            s3 = float(request.form['s3_input'])
            s4 = float(request.form['s4_input'])
            s5 = float(request.form['s5_input'])
            s6 = float(request.form['s6_input'])
            data_test = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
            weight = np.load('static/assets/model/model_linreg.npy')
            stochasticGD = linearregressionmanual.ManualLinearRegression()
            predict = stochasticGD.testing(data_test, weight)
            predict = np.around(predict, decimals=2)
            return render_template('task_linear_regression.html', predict=predict[0])

    else:
        return render_template('task_linear_regression.html')

@app.route('/task_logistic_regression', methods=['GET', 'POST'])
def task_logistic_regression_route():
    if request.method == 'POST':
        mode = request.form['mode']
        if mode == "Training":
            test_size = (100 - int(request.form['komposisi'])) / 100
            learning_rate = float(request.form['learning_rate'])
            epochs = int(request.form['query_epochs'])
            option_show_training = request.form['query_tampil_training']
            data, target = load_diabetes(return_X_y=True)
            data_training, data_test, target_training, target_test = train_test_split(data, target, 
                test_size=test_size, random_state=0)

            data_training = normalize(data_training)
            data_test = normalize(data_test)

            # Deklarasi objects
            logisticRegression = logisticRegression.ManualLogisticRegression()
            weight = logisticRegression.training(data_training, target_training, epochs, learning_rate)
            weight = np.around(weight, decimals=2)
            num_of_weight, dim = weight.shape
            model = 'y = ' + str(round(weight[num_of_weight - 1][0], 2))
            for index_weight in range(1, dim):
                w = round(weight[num_of_weight - 1][index_weight], 2)
                model = model + (' + ' + str(w) + (' * x%s'%(index_weight)))
            predict = logisticRegression.testing(data_test, w[len(w)-1])
            precision, recall = logisticRegression.accurate(target_test, predict)
            if option_show_training == 'Tidak':
                return render_template('task_logistic_regression.html', model=model)
            elif option_show_training == 'Ya':
                return render_template('task_logistic_regression.html', model=model, step_trains=weight, epochs=epochs)
        else:
            return render_template('task_logistic_regression.html')
    else:
        return render_template('task_logistic_regression.html')

@app.route('/deteksi_cnn',methods=['GET', 'POST'])
def deteksi_cnn():
    path_hasil = 'static/assets/deteksi_cnn/hasil/'
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = 'static/assets/deteksi_cnn/upload/'+ file.filename
        result_name = file.filename[:-4]
        img.save(uploaded_img_path)
        img_input = cv2.imread(uploaded_img_path)
        gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path_hasil + result_name +'.jpg', gray)
        return render_template('deteksi_cnn.html',query_path=uploaded_img_path,img=result_name)
    else:
        return render_template('deteksi_cnn.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
