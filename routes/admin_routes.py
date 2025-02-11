from flask import Blueprint, render_template, redirect, request, url_for, flash, session, jsonify
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense, LSTM
from tensorflow.keras.saving import register_keras_serializable
from models import SubmitData, Admin, PerformanceMulti, PerformanceSingle, db
import pandas as pd
import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import io, json
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from plotting import update_chart_10_11_12, update_chart_TN_TN, update_chart_TN_XH
from keras.callbacks import History
import shutil
import os



@register_keras_serializable()
# Customize Model
class MLPModel1(Model):
    def __init__(self):
        super(MLPModel1, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(62, activation='relu') 
        self.fc3 = Dense(32, activation='relu')
        self.fc4 = Dense(18, activation='linear') 
    def call(self, inputs):
        features = self.flatten(inputs)
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.fc3(features)
        features = self.fc4(features)
        return features
    


    
    def get_config(self):
        config = super(MLPModel1, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()
    
@register_keras_serializable()
class MLPModel2(Model):
    def __init__(self):
        super(MLPModel2, self).__init__()
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(62, activation='relu') 
        self.fc3 = Dense(32, activation='relu')
        self.fc4 = Dense(6, activation='linear') 
    def call(self, inputs):
        features = self.flatten(inputs)
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.fc3(features)
        features = self.fc4(features)
        return features
    
    def get_config(self):
        config = super(MLPModel2, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()

@register_keras_serializable()
class LSTMModel1(Model):
    def __init__(self):
        super(LSTMModel1, self).__init__()
        self.LSTM1 = LSTM(128, activation = 'relu', return_sequences = True)
        self.LSTM2 = LSTM(64, activation = 'relu', return_sequences= True)
        self.LSTM3 = LSTM(32, activation = 'relu', return_sequences= False)
        self.fc1 = Dense(18, activation = 'linear')


    def call(self, inputs):
        features = self.LSTM1(inputs)
        features = self.LSTM2(features)
        features = self.LSTM3(features)
        features = self.fc1(features)
        return features
  
    def get_config(self):
        config = super(LSTMModel1, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()
    
@register_keras_serializable()
class LSTMModel2(Model):
    def __init__(self):
        super(LSTMModel2, self).__init__()
        self.LSTM1 = LSTM(128, activation = 'relu', return_sequences = True)
        self.LSTM2 = LSTM(64, activation = 'relu', return_sequences= True)
        self.LSTM3 = LSTM(32, activation = 'relu', return_sequences= False)
        self.fc1 = Dense(6, activation = 'linear')


    def call(self, inputs):
        features = self.LSTM1(inputs)
        features = self.LSTM2(features)
        features = self.LSTM3(features)
        features = self.fc1(features)
        return features
  
    def get_config(self):
        config = super(LSTMModel2, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()
    


admin_blueprint = Blueprint("admin", __name__, template_folder = '../templates/admin')


@admin_blueprint.route('/get_submit_data', methods=['GET'])
def get_submit_data():
    try:
        submit_data = SubmitData.query.first()
        today = date.today()

        if not submit_data:
            return jsonify({
                'total_submit': 0,
                'today_submit': 0
            })
        
        # Reset số lần submit trong ngày (trường hợp ngày mới nhưng chưa reset)
        if submit_data.last_day_submit.date() != today:
            submit_data.submits_today = 0
            submit_data.last_day_submit = today
            db.session.commit()

        return jsonify({
            'total_submit': submit_data.total_submits,
            'today_submit': submit_data.submits_today
        })

    except Exception as e:  
        print("Error occurred:", e)
        return jsonify({'error': 'Internal server error'}), 500
    

@admin_blueprint.route('/getperformance/multi/average', methods=['GET'])
def get_performance_multi_average():
    multi_averages = db.session.query(
        func.avg(PerformanceMulti.latency).label("avg_latency"),
        func.avg(PerformanceMulti.throughput).label("avg_throughput"),
        func.avg(PerformanceMulti.cpu_usage).label("avg_cpu_usage"),
        func.avg(PerformanceMulti.memory_usage).label("avg_memory_usage"),
        func.avg(PerformanceMulti.total_predictions).label("avg_total_predictions")
    ).first()
    return jsonify({
        "avg_latency": round(multi_averages.avg_latency, 2) if multi_averages.avg_latency else 0,
        "avg_throughput": round(multi_averages.avg_throughput, 2) if multi_averages.avg_throughput else 0,
        "avg_cpu_usage": round(multi_averages.avg_cpu_usage, 2) if multi_averages.avg_cpu_usage else 0,
        "avg_memory_usage": round(multi_averages.avg_memory_usage, 2) if multi_averages.avg_memory_usage else 0,
        "avg_total_predictions": round(multi_averages.avg_total_predictions, 2) if multi_averages.avg_total_predictions else 0
    })

@admin_blueprint.route('/getperformance/single/average', methods=['GET'])
def get_performance_single_average():
    single_averages = db.session.query(
        func.avg(PerformanceSingle.latency).label("avg_latency"),
        func.avg(PerformanceSingle.throughput).label("avg_throughput"),
        func.avg(PerformanceSingle.cpu_usage).label("avg_cpu_usage"),
        func.avg(PerformanceSingle.memory_usage).label("avg_memory_usage")
    ).first()
    return jsonify({
        "avg_latency": round(single_averages.avg_latency, 2) if single_averages.avg_latency else 0,
        "avg_throughput": round(single_averages.avg_throughput, 2) if single_averages.avg_throughput else 0,
        "avg_cpu_usage": round(single_averages.avg_cpu_usage, 2) if single_averages.avg_cpu_usage else 0,
        "avg_memory_usage": round(single_averages.avg_memory_usage, 2) if single_averages.avg_memory_usage else 0
    })

    

    
@admin_blueprint.route('/page_admin')
def pageAdmin():
    return render_template('PageAdmin.html')


@admin_blueprint.route('/page_dashboard')
def pageDashBoard():
    return render_template('PageDashBoard.html')


@admin_blueprint.route('/page_change_password')
def pageChangePassword():
    return render_template('PageChangePassword.html')



@admin_blueprint.route("/", strict_slashes=False)
def home():
    return render_template("PageDashBoard.html")



@admin_blueprint.before_request
def require_login():
    if request.endpoint == "admin.admin_login" and session.get("admin_logged_in"):
        return redirect(url_for("admin.pageDashBoard"))
    
    if request.endpoint == "admin.admin_get_submit_data" or request.endpoint == "admin.admin_login":
        return  # Bỏ qua middleware cho những trang này

    if not session.get("admin_logged_in"):
        print("chuyển hướng tới login")
        return redirect(url_for("admin.admin_login"))




@admin_blueprint.route("login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            session["admin_logged_in"] = True
            session["admin_username"] = username
            session.permanent = True
            return redirect(url_for("admin.admin_dashboard"))
        else:
            print(admin.password)
            print("sai mk")
            return render_template("PageLogin.html")
    return render_template("PageLogin.html")




@admin_blueprint.route("change_password", methods=["GET", "POST"])
def change_password():
    if request.method == "POST":
        # Lấy thông tin 
        current_password = request.form["current_password"]
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]

        # Kiểm tra admin 
        admin = Admin.query.filter_by(username=session.get("admin_username")).first()

        # Xác minh mật khẩu
        if not admin or not admin.check_password(current_password):
            return render_template(
                "PageChangePassword.html",
                current_password_error="Sai mật khẩu."
            )

        if new_password != confirm_password:
            return render_template(
                "PageChangePassword.html",
                confirm_password_error="Mật khẩu không khớp."
            )

        # Lưu lại mật khẩu
        admin.set_password(new_password)
        db.session.commit()
        flash("Đổi mật khẩu thành công!", "success")
        return redirect(url_for("admin.admin_logout"))

    return render_template("PageChangePassword.html")




@admin_blueprint.route("dashboard")
def admin_dashboard():
    return render_template("PageDashBoard.html")


@admin_blueprint.route("logout", methods=["POST"])
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for('admin.admin_login'))


def err(subject, actual, pred): 
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    res = {'Subject': subject,'MAE': mae, 'MAPE': mape*100, 'MSE': mse, 'RMSE': rmse, 'Accuracy': 100 - mape*100}
    return res

def Accuracy12(result): # Tính toán độ chính xác cho các mô hình học máy với dữ liệu đầu vào nhận từ hàm result bên dưới và tính toán bằng cách sử dụng các loss function ở hàm err bên trên như một công thức
    Accuracy = pd.DataFrame(columns=['Subject', 'MAE', 'MAPE', 'MSE', 'RMSE', 'Accuracy'])
    data = []
    data.append(err('Maths_1_12', result['Maths_1_12'], result['Maths_1_12_pred'])) #gọi hàm err để tính toán độ chính xác với tham số đầu vào là kết quả thực (Math) và kết quả model dự đoán được (Maths_pred)
    data.append(err('Literature_1_12', result['Literature_1_12'], result['Literature_1_12_pred']))
    data.append(err('Physics_1_12', result['Physics_1_12'], result['Physics_1_12_pred']))
    data.append(err('Chemistry_1_12', result['Chemistry_1_12'], result['Chemistry_1_12_pred']))
    data.append(err('Biology_1_12', result['Biology_1_12'], result['Biology_1_12_pred']))
    data.append(err('History_1_12', result['History_1_12'], result['History_1_12_pred']))
    data.append(err('Geography_1_12', result['Geography_1_12'], result['Geography_1_12_pred']))
    data.append(err('English_1_12', result['English_1_12'], result['English_1_12_pred']))
    data.append(err('Civic Education_1_12', result['Civic Education_1_12'], result['Civic Education_1_12_pred']))

    data.append(err('Maths_2_12', result['Maths_2_12'], result['Maths_2_12_pred'])) #gọi hàm err để tính toán độ chính xác với tham số đầu vào là kết quả thực (Math) và kết quả model dự đoán được (Maths_pred)
    data.append(err('Literature_2_12', result['Literature_2_12'], result['Literature_2_12_pred']))
    data.append(err('Physics_2_12', result['Physics_2_12'], result['Physics_2_12_pred']))
    data.append(err('Chemistry_2_12', result['Chemistry_2_12'], result['Chemistry_2_12_pred']))
    data.append(err('Biology_2_12', result['Biology_2_12'], result['Biology_2_12_pred']))
    data.append(err('History_2_12', result['History_2_12'], result['History_2_12_pred']))
    data.append(err('Geography_2_12', result['Geography_2_12'], result['Geography_2_12_pred']))
    data.append(err('English_2_12', result['English_2_12'], result['English_2_12_pred']))
    data.append(err('Civic Education_2_12', result['Civic Education_2_12'], result['Civic Education_2_12_pred']))
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    Accuracy = pd.concat(dfs, ignore_index=True) #Ghi các giá trị tính toán được vào biến Accuracy

    return Accuracy


def Accuracy_TN(result): 
    Accuracy = pd.DataFrame(columns=['Subject', 'MAE', 'MAPE', 'MSE', 'RMSE', 'Accuracy'])
    data = []
    data.append(err('Maths', result['Maths'], result['Maths_pred'])) #gọi hàm err để tính toán độ chính xác với tham số đầu vào là kết quả thực (Math) và kết quả model dự đoán được (Maths_pred)
    data.append(err('Literature', result['Literature'], result['Literature_pred']))
    data.append(err('Physics', result['Physics'], result['Physics_pred']))
    data.append(err('Chemistry', result['Chemistry'], result['Chemistry_pred']))
    data.append(err('Biology', result['Biology'], result['Biology_pred']))
    data.append(err('English', result['English'], result['English_pred']))
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    Accuracy = pd.concat(dfs, ignore_index=True) #Ghi các giá trị tính toán được vào biến Accuracy

    return Accuracy

def Accuracy_XH(result): 
    Accuracy = pd.DataFrame(columns=['Subject', 'MAE', 'MAPE', 'MSE', 'RMSE', 'Accuracy'])
    data = []
    data.append(err('Maths', result['Maths'], result['Maths_pred'])) # Gọi hàm err để tính toán độ chính xác với tham số đầu vào là kết quả thực (Math) và kết quả model dự đoán được (Maths_pred)
    data.append(err('Literature', result['Literature'], result['Literature_pred']))
    data.append(err('History', result['History'], result['History_pred']))
    data.append(err('Geography', result['Geography'], result['Geography_pred']))
    data.append(err('Civic Education', result['Civic Education'], result['Civic Education_pred']))
    data.append(err('English', result['English'], result['English_pred']))
    dfs = [pd.DataFrame(d, index=[0]) for d in data]
    Accuracy = pd.concat(dfs, ignore_index=True) # Ghi các giá trị tính toán được vào biến Accuracy

    return Accuracy


def AverageAccuracy(accuracy_df):
    # Xác định các cột cần tính trung bình
    metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'Accuracy']

    # Tính giá trị trung bình
    averages = accuracy_df[metrics].mean().round(2)

    # Đặt tên cho kết quả
    averages.name = "Average Accuracy"

    return averages


def result12(ytest, ypred): # Sử dụng như một nơi để lưu dữ liệu, hiển thị thông tin giữa điểm thực tế - điểm dự đoán được
    result = pd.DataFrame({
        'Maths_1_12': pd.Series(ytest[:,0]), #lấy tất cả các hàng (tương ứng với tất cả học sinh), và cột thứ 1 (tương ứng với cột điểm môn toán) sau đó gán tất cả cho Maths để tạo dataframe
        'Maths_1_12_pred': pd.Series(ypred[:, 0]),
        'Literature_1_12': pd.Series(ytest[:,1]),
        'Literature_1_12_pred': pd.Series(ypred[:,1 ]),
        'Physics_1_12': pd.Series(ytest[:,2]),
        'Physics_1_12_pred': pd.Series(ypred[:,2 ]),
        'Chemistry_1_12': pd.Series(ytest[:,3]),
        'Chemistry_1_12_pred': pd.Series(ypred[:, 3]),
        'Biology_1_12': pd.Series(ytest[:,4]),
        'Biology_1_12_pred': pd.Series(ypred[:, 4]),
        'History_1_12': pd.Series(ytest[:,5]),
        'History_1_12_pred': pd.Series(ypred[:, 5]),
        'Geography_1_12': pd.Series(ytest[:,6]),
        'Geography_1_12_pred': pd.Series(ypred[:, 6]),
        'English_1_12': pd.Series(ytest[:,7]),
        'English_1_12_pred': pd.Series(ypred[:, 7]),
        'Civic Education_1_12': pd.Series(ytest[:,8]),
        'Civic Education_1_12_pred': pd.Series(ypred[:, 8]),

        'Maths_2_12': pd.Series(ytest[:,9]),
        'Maths_2_12_pred': pd.Series(ypred[:, 9]),
        'Literature_2_12': pd.Series(ytest[:,10]),
        'Literature_2_12_pred': pd.Series(ypred[:,10 ]),
        'Physics_2_12': pd.Series(ytest[:,11]),
        'Physics_2_12_pred': pd.Series(ypred[:,11 ]),
        'Chemistry_2_12': pd.Series(ytest[:,12]),
        'Chemistry_2_12_pred': pd.Series(ypred[:, 12]),
        'Biology_2_12': pd.Series(ytest[:,13]),
        'Biology_2_12_pred': pd.Series(ypred[:, 13]),
        'History_2_12': pd.Series(ytest[:,14]),
        'History_2_12_pred': pd.Series(ypred[:, 14]),
        'Geography_2_12': pd.Series(ytest[:,15]),
        'Geography_2_12_pred': pd.Series(ypred[:, 15]),
        'English_2_12': pd.Series(ytest[:,16]),
        'English_2_12_pred': pd.Series(ypred[:, 16]),
        'Civic Education_2_12': pd.Series(ytest[:,17]),
        'Civic Education_2_12_pred': pd.Series(ypred[:, 17])})
    for col in result.columns: result[col] = result[col].apply(lambda x: 10 if x > 10 else x)
    return result

def result_TN(ytest, ypred):
    if ytest is None or ypred is None:
        raise ValueError("ytest hoặc ypred không được phép là None.")
    if len(ytest) != len(ypred):
        raise ValueError("Kích thước của ytest và ypred không khớp.")
    
    subjects = ["Maths", "Literature", "Physics", "Chemistry", "Biology", "English"]
    data = {f"{subj}": ytest[:, i] for i, subj in enumerate(subjects)}
    data.update({f"{subj}_pred": ypred[:, i] for i, subj in enumerate(subjects)})
    
    result = pd.DataFrame(data)
    result = result.applymap(lambda x: min(x, 10))  # Giới hạn giá trị về tối đa là 10
    return result


def result_XH(ytest, ypred): # Sử dụng như một nơi để lưu dữ liệu, hiển thị thông tin giữa điểm thực tế - điểm dự đoán được
    result = pd.DataFrame({
        'Maths': pd.Series(ytest[:,0]), # Lấy tất cả các hàng (tương ứng với tất cả học sinh), và cột thứ 1 (tương ứng với cột điểm môn toán) sau đó gán tất cả cho Maths để tạo dataframe
        'Maths_pred': pd.Series(ypred[:, 0]),
        'Literature': pd.Series(ytest[:,1]),
        'Literature_pred': pd.Series(ypred[:,1 ]),
        'History': pd.Series(ytest[:,2]),
        'History_pred': pd.Series(ypred[:, 2]),
        'Geography': pd.Series(ytest[:,3]),
        'Geography_pred': pd.Series(ypred[:, 3]),
        'Civic Education': pd.Series(ytest[:, 4]),
        'Civic Education_pred': pd.Series(ypred[:, 4]),
        'English': pd.Series(ytest[:,5]),
        'English_pred': pd.Series(ypred[:, 5])})
    for col in result.columns: result[col] = result[col].apply(lambda x: 10 if x > 10 else x)
    return result



@admin_blueprint.route("train_model_12", methods=['GET', 'POST'])
def train_model_12():
    if request.method == "POST":
        try:
            # Đọc file test
            excel_file_test = 'E:/Download/ScorePredict + MLOps/data/10_11_12_Test.xlsx'
            dftest = pd.read_excel(excel_file_test, header=None, skiprows=0)
            dftest = dftest.drop(columns=0)           
            xtest = dftest.iloc[:, :37]
            ytest = dftest.iloc[:, 37:]
            ytest = ytest.to_numpy()

            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df = pd.read_excel(excel_file, header=None, skiprows=0)
            excel_file_original = 'E:/Download/ScorePredict + MLOps/data/10_11_12.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df], ignore_index=True)
            excel_output = 'E:/Download/ScorePredict + MLOps/data/10_11_12_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)
            count_colum = df.shape[0]
            df = df.drop(columns=0)  # Bỏ cột đầu tiên (cột tên học sinh)
            xtrain = df.iloc[:, :37]
            ytrain = df.iloc[:, 37:]
            history = History()
            # Hàm train model
            def load_and_train(model_path, model_type):
                if model_type == 'keras':
                    model = load_model(f"Models/{model_path}")  # Load mô hình Keras
                else:
                    with open(f"Models/{model_path}", 'rb') as model_file:
                        model = joblib.load(model_file)  # Load mô hình với joblib

                # Tạo đường dẫn lưu model sau khi train
                after_train_path = f"{model_path.split('.')[0]}_after_train.{model_path.split('.')[-1]}"
                if "LSTM" in model_path:  # Kiểm tra model LSTM
                    type = 'LSTM'
                    xtrain_timesteps = xtrain.values.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
                    xtest_timesteps = xtest.values.reshape((xtest.shape[0], 1, xtest.shape[1]))
                    
                    ypred_before = model.predict(xtest_timesteps)
                    echohs = count_colum/650*200
                    n = round(echohs)
                    # Train model và lưu lại
                    model.fit(xtrain_timesteps, ytrain, epochs=n, batch_size=128, callbacks = [history])
                    model.save(f"Models/{after_train_path}")

                    # Dự đoán trước và sau khi train
                    ypred_after = model.predict(xtest_timesteps)
                elif "MLP" in model_path:  
                    type = 'MLP'
                    ypred_before = model.predict(xtest)
                    echohs = count_colum/650*200
                    n = round(echohs)
                    model.fit(xtrain, ytrain, epochs=200, batch_size=64, callbacks = [history])
                    model.save(f"Models/{after_train_path}")

                    # Dự đoán trước và sau khi train
                    ypred_after = model.predict(xtest)
                else:  # Linear Regression
                    type = 'LR'
                    ypred_before = model.predict(xtest)
                    model = LinearRegression().fit(xtrain, ytrain)
                    with open(f"Models/{after_train_path}", 'wb') as model_file:
                        joblib.dump(model, model_file)

                    # Dự đoán trước và sau khi train
                    ypred_after = model.predict(xtest)

                # Tính toán độ chính xác
                result_before = result12(ytest, ypred_before)
                accuracy_before = Accuracy12(result_before)
                average_accuracy_before = AverageAccuracy(accuracy_before)

                result_after = result12(ytest, ypred_after)
                excel_output = f'E:/Download/ScorePredict + MLOps/data/{type}_Actual_Pred_10_11_12_update.xlsx'
                result_after.to_excel(excel_output, index=False)
                accuracy_after = Accuracy12(result_after)
                average_accuracy_after = AverageAccuracy(accuracy_after)

                return {
                    "model_path": model_path,
                    "accuracy_before": average_accuracy_before,
                    "accuracy_after": average_accuracy_after,
                }

            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, "LR_10_11_12.pkl", model_type='joblib')
                future2 = executor.submit(load_and_train, "MLP_10_11_12.keras", model_type='keras')
                future3 = executor.submit(load_and_train, "LSTM_10_11_12.keras", model_type='keras')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả so sánh
            return render_template(
                "PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3], 
                type_train = "10_11_12"
            )

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    # Hiển thị trang upload file
    return render_template("PageAdminTrainDisplay.html")



@admin_blueprint.route("train_model_TN_TN", methods=['GET', 'POST'])
def train_model_TN_TN():
    if request.method == "POST":
        try:
            # Đọc file test
            excel_file_test = 'E:/Download/ScorePredict + MLOps/data/TN_TN_Test.xlsx'
            dftest = pd.read_excel(excel_file_test, header=None, skiprows=0)
            dftest = dftest.drop(columns=0)
            xtest = dftest.iloc[:, :55]
            ytest = dftest.iloc[:, 55:].to_numpy()

            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df = pd.read_excel(excel_file, header=None, skiprows=1)
            excel_file_original = 'E:/Download/ScorePredict + MLOps/data/TN_TN.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df], ignore_index=True)
            excel_output = 'E:/Download/ScorePredict + MLOps/data/TN_TN_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)
            count_colum = df.shape[0]
            df = df.drop(columns=0)  # Bỏ cột đầu tiên (cột tên)
            xtrain = df.iloc[:, :55]
            ytrain = df.iloc[:, 55:]
            history = History()
            # Hàm train model
            def load_and_train(model_path, model_type):
                try:
                    if model_type == 'keras':
                        model = load_model(f"Models/{model_path}")  # Load mô hình Keras
                    else:
                        with open(f"Models/{model_path}", 'rb') as model_file:
                            model = joblib.load(model_file)  # Load mô hình với joblib

                    # Tạo đường dẫn lưu model sau khi train
                    after_train_path = f"Models/{model_path.split('.')[0]}_after_train.{model_path.split('.')[-1]}"

                    if "LSTM" in model_path:  # LSTM model
                        type = 'LSTM'
                        xtrain_timesteps = xtrain.values.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
                        xtest_timesteps = xtest.values.reshape((xtest.shape[0], 1, xtest.shape[1]))
                        ypred_before = model.predict(xtest_timesteps)
                        model.fit(xtrain_timesteps, ytrain, epochs=count_colum, batch_size=128, callbacks = [history])
                        model.save(after_train_path)

                        # Độ chính xác trước và sau khi train
                        ypred_after = model.predict(xtest_timesteps)
                    elif "MLP" in model_path:  # MLP model
                        type = 'MLP'
                        ypred_before = model.predict(xtest)
                        model.fit(xtrain, ytrain, epochs=count_colum, batch_size=64, callbacks = [history])
                        model.save(after_train_path)

                        # Độ chính xác trước và sau khi train
                        ypred_after = model.predict(xtest)
                    else:  # Linear Regression
                        type = 'LR'
                        ypred_before = model.predict(xtest)
                        model = LinearRegression().fit(xtrain, ytrain)
                        with open(after_train_path, 'wb') as model_file:
                            joblib.dump(model, model_file)

                        # Độ chính xác trước và sau khi train
                        ypred_after = model.predict(xtest)

                    # Tính toán độ chính xác trước và sau khi train
                    result_before = result_TN(ytest, ypred_before)
                    accuracy_before = Accuracy_TN(result_before)
                    average_accuracy_before = AverageAccuracy(accuracy_before)

                    result_after = result_TN(ytest, ypred_after)
                    excel_output = f'E:/Download/ScorePredict + MLOps/data/{type}_Actual_Pred_TN_TN_update.xlsx'
                    result_after.to_excel(excel_output, index=False)
                    accuracy_after = Accuracy_TN(result_after)
                    average_accuracy_after = AverageAccuracy(accuracy_after)

                    return {
                        "model_path": model_path,
                        "accuracy_before": average_accuracy_before,
                        "accuracy_after": average_accuracy_after,
                    }
                except Exception as e:
                    print(f"Lỗi khi train mô hình {model_path}: {e}")
                    return {"error": str(e)}

            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, "LR_TN_TN.pkl", model_type='joblib')
                future2 = executor.submit(load_and_train, "MLP_TN_TN.keras", model_type='keras')
                future3 = executor.submit(load_and_train, "LSTM_TN_TN.keras", model_type='keras')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả
            return render_template("PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3], 
                type_train = "TN_TN")

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    return render_template("PageAdminTrainDisplay.html")


@admin_blueprint.route("train_model_TN_XH", methods=['GET', 'POST'])
def train_model_TN_XH():
    if request.method == "POST":
        try:
            # Đọc file test
            excel_file_test = 'E:/Download/ScorePredict + MLOps/data/TN_XH_Test.xlsx'
            dftest = pd.read_excel(excel_file_test, header=None)
            dftest =  dftest.drop(columns=0)
            xtest = dftest.iloc[:, :55]
            ytest = dftest.iloc[:, 55:]
            ytest = ytest.to_numpy()

            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df = pd.read_excel(excel_file, header=None, skiprows=1)
            excel_file_original = 'E:/Download/ScorePredict + MLOps/data/TN_XH.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df], ignore_index=True)
            excel_output = 'E:/Download/ScorePredict + MLOps/data/TN_XH_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)
            count_colum = df.shape[0]
            df = df.drop(columns=0)  # Bỏ cột đầu tiên (cột tên)
            xtrain = df.iloc[:, :55]
            ytrain = df.iloc[:, 55:]
            history = History()
            # Hàm train model
            def load_and_train(model_path, model_type):
                if model_type == 'keras':
                    model = load_model(f"Models/{model_path}")  # Load mô hình Keras
                else:
                    with open(f"Models/{model_path}", 'rb') as model_file:
                        model = joblib.load(model_file)  # Load mô hình với joblib

                # Tạo đường dẫn lưu model sau khi train
                after_train_path = f"Models/{model_path.split('.')[0]}_after_train.{model_path.split('.')[-1]}"

                if "LSTM" in model_path:  # LSTM model
                    type = 'LSTM'
                    xtrain_timesteps = xtrain.values.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
                    xtest_timesteps = xtest.values.reshape((xtest.shape[0], 1, xtest.shape[1]))
                    echohs = count_colum/2
                    n = round(echohs)
                    ypred_before = model.predict(xtest_timesteps)
                    model.fit(xtrain_timesteps, ytrain, epochs=n, batch_size=128, callbacks = [history])
                    model.save(after_train_path)

                    # Độ chính xác trước và sau khi train
                    ypred_after = model.predict(xtest_timesteps)
                elif "MLP" in model_path:  # MLP model
                    type = 'MLP'
                    echohs = count_colum/2
                    n = round(echohs)
                    ypred_before = model.predict(xtest)
                    model.fit(xtrain, ytrain, epochs=n, batch_size=64, callbacks = [history])
                    model.save(after_train_path)

                    # Độ chính xác trước và sau khi train
                    ypred_after = model.predict(xtest)
                else:  # Linear Regression
                    type = 'LR'
                    ypred_before = model.predict(xtest)
                    model = LinearRegression().fit(xtrain, ytrain)
                    with open(after_train_path, 'wb') as model_file:
                        joblib.dump(model, model_file)

                    # Độ chính xác trước và sau khi train
                    ypred_after = model.predict(xtest)
                # Tính toán độ chính xác
                result_before = result_XH(ytest, ypred_before)
                accuracy_before = Accuracy_XH(result_before)
                average_accuracy_before = AverageAccuracy(accuracy_before)

                result_after = result_XH(ytest, ypred_after)
                accuracy_after = Accuracy_XH(result_after)
                excel_output = f'E:/Download/ScorePredict + MLOps/data/{type}_Actual_Pred_TN_XH_update.xlsx'
                result_after.to_excel(excel_output, index=False)
                average_accuracy_after = AverageAccuracy(accuracy_after)

                return {
                    "model_path": model_path,
                    "accuracy_before": average_accuracy_before,
                    "accuracy_after": average_accuracy_after,
                }

            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, "LR_TN_XH.pkl", model_type='joblib')
                future2 = executor.submit(load_and_train, "MLP_TN_XH.keras", model_type='keras')
                future3 = executor.submit(load_and_train, "LSTM_TN_XH.keras", model_type='keras')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả
            return render_template(
                "PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3],
                type_train = "TN_XH"
            )

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    # Hiển thị trang upload file
    return render_template("PageAdminTrainDisplay.html")




@admin_blueprint.route("accept_trained_models", methods=['POST'])
def accept_trained_models():
    try:
        type_train = request.form.get('type_train')
        if not type_train:
            return {"error": "Missing 'type_train' parameter"}, 400

        trained_model_paths = [
            f"models/LR_{type_train}_after_train.pkl",
            f"models/MLP_{type_train}_after_train.keras",
            f"models/LSTM_{type_train}_after_train.keras"
        ]

        data_original_update_paths = f"data/{type_train}_update.xlsx"

        data_pred_update_paths = [
            f"data/LR_Actual_Pred_{type_train}_update.xlsx",
            f"data/MLP_Actual_Pred_{type_train}_update.xlsx",
            f"data/LSTM_Actual_Pred_{type_train}_update.xlsx"
        ]

        for path in trained_model_paths:
            original_path = path.replace("_after_train", "")  # Model chính thức
            shutil.move(path, original_path)  # Ghi đè model cũ bằng model mới

        original_path = data_original_update_paths.replace("_update", "")  # Thay "_update" bằng tên gốc
        shutil.move(data_original_update_paths, original_path)  # Di chuyển tệp  # Ghi đè dữ liệu đã train cũ bằng dữ liệu mới

        for path in data_pred_update_paths:
            original_path = path.replace("_update", "") # Ghi đè dữ liệu tính cũ bằng dữ liệu mới
            shutil.move(path, original_path)
        
        if type_train == "10_11_12":
            update_chart_10_11_12()
        elif type_train == "TN_TN":
            update_chart_TN_TN()
        else:
            update_chart_TN_XH()

        return {"message": "Model sau khi train đã được chấp nhận và lưu!"}

    except Exception as e:
        print(f"Lỗi khi chấp nhận model: {e}")
        return {"error": str(e)}, 500



@admin_blueprint.route("reject_trained_models", methods=['POST'])
def reject_trained_models():
    try:
        # Xóa các model đã train
        type_train = request.form.get('type_train')
        if not type_train:
            return {"error": "Missing 'type_train' parameter"}, 400

        # Xóa các model đã train

        trained_model_paths = [
            f"Models/LR_{type_train}_after_train.pkl",
            f"Models/MLP_{type_train}_after_train.keras",
            f"Models/LSTM_{type_train}_after_train.keras"
        ]
        data_original_update_paths = [
            f"data/{type_train}_update.xlsx",
            f"data/{type_train}_update.xlsx",
            f"data/{type_train}_update.xlsx"
        ]
        data_pred_update_paths = [
            f"data/LR_Actual_Pred_{type_train}_update.xlsx",
            f"data/MLP_Actual_Pred_{type_train}_update.xlsx",
            f"data/LSTM_Actual_Pred_{type_train}_update.xlsx"
        ]



        for path in trained_model_paths:
            print(path)
            if os.path.exists(path):
                os.remove(path)  # Xóa model đã train

        for path in data_original_update_paths:
            print(path)
            if os.path.exists(path):
                os.remove(path)  # Xóa dữ liệu train mới

        for path in data_pred_update_paths:
            print(path)
            if os.path.exists(path):
                os.remove(path)  # Xóa dữ liệu tính toán mới

        return redirect(url_for("admin.pageDashBoard"))
    except Exception as e:
        print(f"Lỗi khi từ chối model: {e}")
        return {"error": str(e)}, 500