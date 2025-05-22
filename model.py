#model.py
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense, LSTM
from tensorflow.keras.saving import register_keras_serializable
from database import SubmitData, Admin, PerformanceMulti, PerformanceSingle, db
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
from keras.callbacks import History
from pathlib import Path
from sklearn.model_selection import train_test_split






@register_keras_serializable()
# Customize Model
class MLPModel12(Model):
    def __init__(self):
        super(MLPModel12, self).__init__()
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
        config = super(MLPModel12, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()



@register_keras_serializable()
# Customize Model
class MLPModelTN(Model):
    def __init__(self):
        super(MLPModelTN, self).__init__()
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
        config = super(MLPModelTN, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()

@register_keras_serializable()
class LSTMModel12(Model):
    def __init__(self):
        super(LSTMModel12, self).__init__()
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
        config = super(LSTMModel12, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()
    


@register_keras_serializable()
class LSTMModelTN(Model):
    def __init__(self):
        super(LSTMModelTN, self).__init__()
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
        config = super(LSTMModelTN, self).get_config()
        # Return the config dictionary including any custom parameters
        return config

    @classmethod
    def from_config(cls, config):
        # Create a new instance of the model from the config
        return cls()
    



subject_12 = ['Name',
         'Maths_1_10', 'Literature_1_10', 'Physics_1_10', 'Chemistry_1_10', 'Biology_1_10', 'History_1_10', 'Geography_1_10', 'English_1_10', 'Civic Education_1_10',
         'Maths_2_10', 'Literature_2_10', 'Physics_2_10', 'Chemistry_2_10', 'Biology_2_10', 'History_2_10', 'Geography_2_10', 'English_2_10', 'Civic Education_2_10',
         'Maths_1_11', 'Literature_1_11', 'Physics_1_11', 'Chemistry_1_11', 'Biology_1_11', 'History_1_11', 'Geography_1_11', 'English_1_11', 'Civic Education_1_11',
         'Maths_2_11', 'Literature_2_11', 'Physics_2_11', 'Chemistry_2_11', 'Biology_2_11', 'History_2_11', 'Geography_2_11', 'English_2_11', 'Civic Education_2_11',
         'Maths_1_12', 'Literature_1_12', 'Physics_1_12', 'Chemistry_1_12', 'Biology_1_12', 'History_1_12', 'Geography_1_12', 'English_1_12', 'Civic Education_1_12',
         'Maths_2_12', 'Literature_2_12', 'Physics_2_12', 'Chemistry_2_12', 'Biology_2_12', 'History_2_12', 'Geography_2_12', 'English_2_12', 'Civic Education_2_12',
         'orphan_and_kios',
         'Maths_1_12', 'Literature_1_12', 'Physics_1_12', 'Chemistry_1_12', 'Biology_1_12', 'History_1_12', 'Geography_1_12', 'English_1_12', 'Civic Education_1_12',
         'Maths_2_12', 'Literature_2_12', 'Physics_2_12', 'Chemistry_2_12', 'Biology_2_12', 'History_2_12', 'Geography_2_12', 'English_2_12', 'Civic Education_2_12',
         ]


subject_TN = ['Name',
         'Maths_1_10', 'Literature_1_10', 'Physics_1_10', 'Chemistry_1_10', 'Biology_1_10', 'History_1_10', 'Geography_1_10', 'English_1_10', 'Civic Education_1_10',
         'Maths_2_10', 'Literature_2_10', 'Physics_2_10', 'Chemistry_2_10', 'Biology_2_10', 'History_2_10', 'Geography_2_10', 'English_2_10', 'Civic Education_2_10',
         'Maths_1_11', 'Literature_1_11', 'Physics_1_11', 'Chemistry_1_11', 'Biology_1_11', 'History_1_11', 'Geography_1_11', 'English_1_11', 'Civic Education_1_11',
         'Maths_2_11', 'Literature_2_11', 'Physics_2_11', 'Chemistry_2_11', 'Biology_2_11', 'History_2_11', 'Geography_2_11', 'English_2_11', 'Civic Education_2_11',
         'Maths_1_12', 'Literature_1_12', 'Physics_1_12', 'Chemistry_1_12', 'Biology_1_12', 'History_1_12', 'Geography_1_12', 'English_1_12', 'Civic Education_1_12',
         'Maths_2_12', 'Literature_2_12', 'Physics_2_12', 'Chemistry_2_12', 'Biology_2_12', 'History_2_12', 'Geography_2_12', 'English_2_12', 'Civic Education_2_12',
         'orphan_and_kios',
         'Maths_TN', 'Literature_TN', 'Physics_TN', 'Chemistry_TN', 'Biology_TN', 'English_TN']

subject_XH = ['Name',
         'Maths_1_10', 'Literature_1_10', 'Physics_1_10', 'Chemistry_1_10', 'Biology_1_10', 'History_1_10', 'Geography_1_10', 'English_1_10', 'Civic Education_1_10',
         'Maths_2_10', 'Literature_2_10', 'Physics_2_10', 'Chemistry_2_10', 'Biology_2_10', 'History_2_10', 'Geography_2_10', 'English_2_10', 'Civic Education_2_10',
         'Maths_1_11', 'Literature_1_11', 'Physics_1_11', 'Chemistry_1_11', 'Biology_1_11', 'History_1_11', 'Geography_1_11', 'English_1_11', 'Civic Education_1_11',
         'Maths_2_11', 'Literature_2_11', 'Physics_2_11', 'Chemistry_2_11', 'Biology_2_11', 'History_2_11', 'Geography_2_11', 'English_2_11', 'Civic Education_2_11',
         'Maths_1_12', 'Literature_1_12', 'Physics_1_12', 'Chemistry_1_12', 'Biology_1_12', 'History_1_12', 'Geography_1_12', 'English_1_12', 'Civic Education_1_12',
         'Maths_2_12', 'Literature_2_12', 'Physics_2_12', 'Chemistry_2_12', 'Biology_2_12', 'History_2_12', 'Geography_2_12', 'English_2_12', 'Civic Education_2_12',
         'orphan_and_kios',
         'Maths_TN', 'Literature_TN', 'History_TN', 'Geography_TN', 'Civic Education_TN', 'English_TN']

subject_result_12 = ['Maths_1_12', 'Literature_1_12', 'Physics_1_12', 'Chemistry_1_12', 'Biology_1_12', 'History_1_12', 'Geography_1_12', 'English_1_12', 'Civic Education_1_12',
              'Maths_2_12', 'Literature_2_12', 'Physics_2_12', 'Chemistry_2_12', 'Biology_2_12', 'History_2_12', 'Geography_2_12', 'English_2_12', 'Civic Education_2_12',]


subject_result_TN = ['Maths', 'Literature', 'History', 'Geography', 'Civic Education', 'English']


subject_result_XH = ['Maths', 'Literature', 'History', 'Geography', 'Civic Education', 'English']



def err(subject, actual, pred): 
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    res = {'Subject': subject,'MAE': mae, 'MAPE': mape*100, 'MSE': mse, 'RMSE': rmse, 'Accuracy': 100 - mape*100}
    return res


def Accuracy(result, subjects):
    subjects = []
    Accuracy = pd.DataFrame(columns=['Subject', 'MAE', 'MAPE', 'MSE', 'RMSE', 'Accuracy'])
    data = []
    for subject in subjects:
        data.append(err(subject, result[subject], result[f'{subject}_pred'])) # Gọi hàm err để tính toán độ chính xác với tham số đầu vào là kết quả thực (Math) và kết quả model dự đoán được (Maths_pred)
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


def generate_result(ytest, ypred, subjects):
    if ytest is None or ypred is None:
        raise ValueError("ytest hoặc ypred không được phép là None.")
    if len(ytest) != len(ypred):
        raise ValueError("Kích thước của ytest và ypred không khớp.")


            
    data = {f"{subj}": ytest[:, i] for i, subj in enumerate(subjects)}
    data.update({f"{subj}_pred": ypred[:, i] for i, subj in enumerate(subjects)})
    
    result = pd.DataFrame(data)
    result = result.applymap(lambda x: min(x, 10))  # Giới hạn điểm tối đa là 10
    return result



def load_and_predict(school_name, model_path, model_type, pred_arg_arr):
    if model_type == 'keras':
        model = load_model(f"{school_name}Models/{model_path}")  # Load model Keras
    else:
        with open(f"{school_name}Models/{model_path}", 'rb') as model_file:
            model = joblib.load(model_file)  # Load model với joblib
        
    if model_type == 'keras' and "LSTM" in model_path:
        return model.predict(pred_arg_arr.reshape(pred_arg_arr.shape[0], 1, pred_arg_arr.shape[1]))
    else:
        return model.predict(pred_arg_arr)






def load_and_train(school_name, model_path, model_type, df, type_subjects):
    try:
        history = History()
        if type_subjects == '10_11_12': 
            count_colum = 37
            subjects = subject_result_12
        elif type_subjects == 'TN_TN':
            subjects = subject_result_TN
            count_colum = 55
        else:
            subjects = subject_result_XH
            count_colum = 55
        df = df.drop(columns=0)  # Bỏ cột đầu tiên (cột tên học sinh)
        x = df.iloc[:, :count_colum]
        y = df.iloc[:, count_colum:]
        count_colum = df.shape[0]
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state= 42)
        if model_type == 'keras':
            model = load_model(f"Models/{school_name}/{model_path}")  # Load mô hình Keras
        else:
            with open(f"Models/{school_name}/{model_path}", 'rb') as model_file:
                model = joblib.load(model_file)  # Load mô hình với joblib

        # Tạo đường dẫn lưu model sau khi train
        after_train_path = f"Models/{school_name}/{model_path.split('.')[0]}_after_train.{model_path.split('.')[-1]}"

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
        result_before = generate_result(ytest, ypred_before, subjects)
        accuracy_before = Accuracy(result_before, subjects)
        average_accuracy_before = AverageAccuracy(accuracy_before)

        result_after = generate_result(ytest, ypred_after, subjects)
        excel_output = Path(f'Data/{school_name}/') / f'{type}_Actual_Pred_{type_subjects}_update.xlsx'
        result_after.to_excel(excel_output, index=False)
        accuracy_after = Accuracy(result_after, subjects)
        average_accuracy_after = AverageAccuracy(accuracy_after)

        return {
            "model_path": model_path,
            "accuracy_before": average_accuracy_before,
            "accuracy_after": average_accuracy_after,
            }
    except Exception as e:
        print(f"Lỗi khi train mô hình {model_path}: {e}")
        return {"error": str(e)}
    



def train_new(school_name, model_path, df, type_subjects):
    try:    
        history = History()
        folder_path = Path(f"Models/{school_name}")
        folder_path.mkdir(parents=True, exist_ok=True)
        if type_subjects == '10_11_12': 
            count_colum = 37
            subjects = subject_result_12
        elif type_subjects == 'TN_TN':
            subjects = subject_result_TN
            count_colum = 55
        else:
            subjects = subject_result_XH
            count_colum = 55
        df = df.drop(columns=0)  # Bỏ cột đầu tiên (cột tên học sinh)
        x = df.iloc[:, :count_colum]
        y = df.iloc[:, count_colum:]
        count_colum = df.shape[0]
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state= 42)
        if "LSTM" in model_path:  # LSTM model
            type = 'LSTM'
            model_path = f"Models/{school_name}/{type}_{type_subjects}.keras"
            folder_path = Path(f"Models/{school_name}")
            folder_path.mkdir(parents=True, exist_ok=True)
            xtrain_timesteps = xtrain.values.reshape((xtrain.shape[0], 1, xtrain.shape[1]))
            xtest_timesteps = xtest.values.reshape((xtest.shape[0], 1, xtest.shape[1]))
            model = LSTMModel()
            model.fit(xtrain_timesteps, ytrain, epochs=count_colum, batch_size=128, callbacks = [history])
            model.save(model_path)
            ypred = model.predict(xtest_timesteps)
        elif "MLP" in model_path:  # MLP model
            type = 'MLP'
            model_path = f"Models/{school_name}/{type}_{type_subjects}.keras"
            model = MLPModel()
            model.fit(xtrain, ytrain, epochs=count_colum, batch_size=64, callbacks = [history])
            model.save(model_path)
            ypred = model.predict(xtest)
        else:  # Linear Regression
            type = 'LR'
            model_path = f"Models/{school_name}/{type}_{type_subjects}.pkl"
            model = LinearRegression().fit(xtrain, ytrain)
            with open(model_path, 'wb') as model_file:
                joblib.dump(model, model_file)

            ypred = model.predict(xtest)

        result = generate_result(ytest, ypred, subjects)
        excel_output = Path(f'Data/{school_name}/') / f'{type}_Actual_Pred_{type_subjects}_update.xlsx'
        result.to_excel(excel_output, index=False)
        accuracy = Accuracy(result, subjects)
        average_accuracy = AverageAccuracy(accuracy)

        return {
            "model_path": model_path,
            "accuracy_before": average_accuracy,
            }
    except Exception as e:
        print(f"Lỗi khi train mô hình {model_path}: {e}")
        return {"error": str(e)}

