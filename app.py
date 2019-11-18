from flask import Flask, render_template, request
import cv2
from PIL import Image
import linearregressionmanual, logisticregressionmanual
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
@app.route('/')
@app.route('/index')
def hello_world():
	return render_template('index.html')

# @app.after_request
# def add_header(response):
#     '''
#     Add header to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     '''
#     response.headers['X-UA-Compatible'] = 'IE=Edge, chrome=1'
#     response.headers['Cache-Control'] = 'public, max-age=0'
#     return response

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
        epochs = int(request.form['query_epochs'])
        test_size = int(request.form['query_split_data_test']) / 100
        option_show_training = request.form['query_tampil_training']

        # Training
        if epochs != None or test_size != None or option_show_training != None:
            data, target = load_diabetes(return_X_y=True)
            data_training, data_test, target_training, target_test = train_test_split(data, target, 
                test_size=test_size, random_state=0)
            stochasticGD = linearregressionmanual.ManualLinearRegression()
            weight = stochasticGD.training(data_training, target_training, epochs, 0.05)
            weight = np.around(weight, decimals=2)
            num_of_weight, dim = weight.shape
            model = 'y = ' + str(round(weight[num_of_weight - 1][0], 2))
            for index_weight in range(1, dim):
                w = round(weight[num_of_weight - 1][index_weight], 2)
                model = model + (' + ' + str(w) + (' * x%s'%(index_weight)))
            plotting_model(weight, data_training[:, 2], target_training)
            if option_show_training == 'Tidak':
                return render_template('task_linear_regression.html', model=model)
            elif option_show_training == 'Ya':
                return render_template('task_linear_regression.html', model=model, step_trains=weight, epochs=epochs)

    else:
        return render_template('task_linear_regression.html')

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
