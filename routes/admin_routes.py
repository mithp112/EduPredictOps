from flask import Blueprint, render_template, redirect, request, url_for, flash, session, jsonify
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
from plotting import update_chart_10_11_12, update_chart_TN_TN, update_chart_TN_XH
from keras.callbacks import History
import shutil
import os
from model import load_and_train, train_new 
from processingdata import process_student_data
from databaseaction import updateAdmin, verify_admin, get_all_schools_and_years, addSchool, updateSchool, get_performance_single, get_performance_multi, get_performance_single_average, get_performance_multi_average, get_submit_data
from pathlib import Path

   


admin_blueprint = Blueprint("admin", __name__, template_folder = '../templates/admin')


@admin_blueprint.route('/get_submit_data', methods=['GET'])
def get_submit_data():
    result = get_submit_data()
    return jsonify(result)
    
    

@admin_blueprint.route('/get_schools', methods =['GET'])
def get_schools():
    result = get_all_schools_and_years()
    return jsonify(result)
    

@admin_blueprint.route('/getperformance/multi/average', methods=['GET'])
def get_performance_multi_average():
    result = get_performance_multi_average()
    return jsonify(result)

@admin_blueprint.route('/getperformance/single/average', methods=['GET'])
def get_performance_single_average():
    result = get_performance_single_average()
    return jsonify(result)

@admin_blueprint.route('/getperformance/multi', methods=['GET'])
def get_performance_multi():
    result = get_performance_multi()
    return jsonify(result)

@admin_blueprint.route('/getperformance/single', methods=['GET'])
def get_performance_single():
    result = get_performance_single()
    return jsonify(result)


    

    
@admin_blueprint.route('/page_admin')
def pageAdmin():
    return render_template('PageAdmin.html')


@admin_blueprint.route('/page_dashboard')
def pageDashBoard():
    return render_template('PageDashBoard.html')


@admin_blueprint.route('/page_change_password')
def pageChangePassword():
    return render_template('PageChangePassword.html')

@admin_blueprint.route('/page_processing_data')
def pageProcessingData():
    return render_template('PageProcessingData.html')

@admin_blueprint.route('/page_success')
def pageSuccess():
    return render_template('PageSuccess.html')

@admin_blueprint.route('/page_review_excel')
def pageReviewExcel():
    return render_template('PageReviewExcel.html')


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
def adminLogin():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        result = verify_admin(username, password)
        if result:
            session["admin_logged_in"] = True
            session["admin_username"] = username
            session.permanent = True
            return redirect(url_for("admin.pageDashBoard"))
        else:
            return render_template("PageLogin.html")
    return render_template("PageLogin.html")




@admin_blueprint.route("change_password", methods=["GET", "POST"])
def change_password():
    if request.method == "POST":
        # Lấy thông tin 
        username=session.get("admin_username")
        current_password = request.form["current_password"]
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]


        if new_password != confirm_password:
            return render_template(
                "PageChangePassword.html",
                confirm_password_error="Mật khẩu không khớp."
            )
        
        result = updateAdmin(username, current_password, new_password)
        # Xác minh mật khẩu
        if not result:
            return render_template(
                "PageChangePassword.html",
                current_password_error="Sai mật khẩu."
            )

        
        flash("Đổi mật khẩu thành công!", "success")
        return redirect(url_for("admin.adminLogout"))

    return render_template("PageChangePassword.html")







@admin_blueprint.route("logout", methods=["POST"])
def adminLogout():
    session.pop("admin_logged_in", None)
    return redirect(url_for('admin.adminLogin'))




@admin_blueprint.route("train_model_12", methods=['GET', 'POST'])
def train_model_12():
    if request.method == "POST":
        try:
            school_name = request.form.get('schoolName')
            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']#note
            df_train = pd.read_excel(excel_file, header=None, skiprows=0)

            excel_file_original = Path(f'Data/{school_name}/') / '10_11_12.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df_train], ignore_index=True)
            excel_output = Path(f'Data/{school_name}/') / '10_11_12_update.xlsx'
            
            df_combine.to_excel(excel_output, index=False, header=False)


            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, school_name,"LR_10_11_12.pkl", model_type='joblib', df = df_train, type_sujects='10_11_12')
                future2 = executor.submit(load_and_train, school_name,"MLP_10_11_12.keras", model_type='keras', df = df_train, type_sujects='10_11_12')
                future3 = executor.submit(load_and_train, school_name,"LSTM_10_11_12.keras", model_type='keras', df = df_train, type_sujects='10_11_12')
                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả so sánh
            return render_template(
                "PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3], 
                type_train = "10_11_12",
                school_name = school_name
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
            school_name = request.form.get('schoolName')
            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df_train = pd.read_excel(excel_file, header=None, skiprows=1)
            excel_file_original = Path(f'Data/{school_name}/') /'TN_TN.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df_train], ignore_index=True)
            excel_output =Path(f'Data/{school_name}/') /'TN_TN_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)
            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, school_name,"LR_TN_TN.pkl", model_type='joblib', df = df_train, type_sujects='TN_TN')
                future2 = executor.submit(load_and_train, school_name,"MLP_TN_TN.keras", model_type='keras', df = df_train, type_sujects='TN_TN')
                future3 = executor.submit(load_and_train, school_name,"LSTM_TN_TN.keras", model_type='keras', df = df_train, type_sujects='TN_TN')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả
            return render_template("PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3], 
                type_train = "TN_TN",
                school_name = school_name)

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    return render_template("PageAdminTrainDisplay.html")


@admin_blueprint.route("train_model_TN_XH", methods=['GET', 'POST'])
def train_model_TN_XH():
    if request.method == "POST":
        try:
            school_name = request.form.get('schoolName')
            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df_train = pd.read_excel(excel_file, header=None, skiprows=1)
            excel_file_original = Path(f'Data/{school_name}/') /'TN_XH.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df_train], ignore_index=True)
            excel_output = Path(f'Data/{school_name}/') /'TN_XH_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)

            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, school_name,"LR_TN_XH.pkl", model_type='joblib', df = df_train, type_sujects='TN_XH')
                future2 = executor.submit(load_and_train, school_name,"MLP_TN_XH.keras", model_type='keras', df = df_train, type_sujects='TN_XH')
                future3 = executor.submit(load_and_train, school_name,"LSTM_TN_XH.keras", model_type='keras', df = df_train, type_sujects='TN_XH')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả
            return render_template(
                "PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3],
                type_train = "TN_XH",
                school_name = school_name
            )

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    # Hiển thị trang upload file
    return render_template("PageAdminTrainDisplay.html")




@admin_blueprint.route("accept_trained_models", methods=['POST'])
def accept_trained_models():
    try:
        school_name = request.form.get('schoolName')
        type_train = request.form.get('type_train')
        if not type_train:
            return {"error": "Missing 'type_train' parameter"}, 400

        trained_model_paths = [
            f"models/{school_name}/LR_{type_train}_after_train.pkl",
            f"models/{school_name}/MLP_{type_train}_after_train.keras",
            f"models/{school_name}/LSTM_{type_train}_after_train.keras"
        ]

        data_original_update_paths = f"data/{type_train}_update.xlsx"

        data_pred_update_paths = [
            f"data/{school_name}/LR_Actual_Pred_{type_train}_update.xlsx",
            f"data/{school_name}/MLP_Actual_Pred_{type_train}_update.xlsx",
            f"data/{school_name}/LSTM_Actual_Pred_{type_train}_update.xlsx"
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
        school_name = request.form.get('schoolName')
        # Xóa các model đã train
        type_train = request.form.get('type_train')
        if not type_train:
            return {"error": "Missing 'type_train' parameter"}, 400

        # Xóa các model đã train

        trained_model_paths = [
            f"Models/{school_name}/LR_{type_train}_after_train.pkl",
            f"Models/{school_name}/MLP_{type_train}_after_train.keras",
            f"Models/{school_name}/LSTM_{type_train}_after_train.keras"
        ]
        data_original_update_paths = [
            f"data/{school_name}/{type_train}_update.xlsx",
            f"data/{school_name}/{type_train}_update.xlsx",
            f"data/{school_name}/{type_train}_update.xlsx"
        ]
        data_pred_update_paths = [
            f"data/{school_name}/LR_Actual_Pred_{type_train}_update.xlsx",
            f"data/{school_name}/MLP_Actual_Pred_{type_train}_update.xlsx",
            f"data/{school_name}/LSTM_Actual_Pred_{type_train}_update.xlsx"
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
    



@admin_blueprint.route("add_data_existed", methods=['POST'])
def data_processing_existed():
    if request.method == 'POST':
        try:
            school_name = request.form['school_name']
            year = float(request.form['year'])
            process_student_data(year, year+2)
            return redirect(url_for("admin.pageSuccess"))
        except Exception as e:
            print(f"Lỗi khi từ chối model: {e}")
            return {"error": str(e)}, 500
        

@admin_blueprint.route("add_data_not_existed", methods=['POST'])
def data_processing_not_existed():
    if request.method == 'POST':
        try:
            start_year = float(request.form['start_year'])
            end_year = float(request.form['end_year'])
            process_student_data(end_year, start_year)
            return redirect(url_for("admin.pageSuccess"))
        except Exception as e:
            print(f"Lỗi khi từ chối model: {e}")
            return {"error": str(e)}, 500
        


@admin_blueprint.route("train_new_model_12", methods=['GET', 'POST'])
def train_new_model_12():
    if request.method == "POST":
        try:
            school_name = request.form.get('schoolName')
            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df_train = pd.read_excel(excel_file, header=None, skiprows=0)

            excel_file_original = Path(f'Data/{school_name}/') / '10_11_12.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df_train], ignore_index=True)
            excel_output = Path(f'Data/{school_name}/') / '10_11_12_update.xlsx'
            
            df_combine.to_excel(excel_output, index=False, header=False)


            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, school_name,"LR_10_11_12.pkl", model_type='joblib', df = df_train, type_sujects='10_11_12')
                future2 = executor.submit(load_and_train, school_name,"MLP_10_11_12.keras", model_type='keras', df = df_train, type_sujects='10_11_12')
                future3 = executor.submit(load_and_train, school_name,"LSTM_10_11_12.keras", model_type='keras', df = df_train, type_sujects='10_11_12')
                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả so sánh
            return render_template(
                "PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3], 
                type_train = "10_11_12",
                school_name = school_name
            )

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    # Hiển thị trang upload file
    return render_template("PageAdminTrainDisplay.html")



@admin_blueprint.route("train_new_model_TN_TN", methods=['GET', 'POST'])
def train_new_model_TN_TN():
    if request.method == "POST":
        try:
            school_name = request.form.get('schoolName')
            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df_train = pd.read_excel(excel_file, header=None, skiprows=1)
            excel_file_original = Path(f'Data/{school_name}/') /'TN_TN.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df_train], ignore_index=True)
            excel_output =Path(f'Data/{school_name}/') /'TN_TN_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)
            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, school_name,"LR_TN_TN.pkl", model_type='joblib', df = df_train, type_sujects='TN_TN')
                future2 = executor.submit(load_and_train, school_name,"MLP_TN_TN.keras", model_type='keras', df = df_train, type_sujects='TN_TN')
                future3 = executor.submit(load_and_train, school_name,"LSTM_TN_TN.keras", model_type='keras', df = df_train, type_sujects='TN_TN')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả
            return render_template("PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3], 
                type_train = "TN_TN",
                school_name = school_name)

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    return render_template("PageAdminTrainDisplay.html")


@admin_blueprint.route("train_new_model_TN_XH", methods=['GET', 'POST'])
def train_new_model_TN_XH():
    if request.method == "POST":
        try:
            school_name = request.form.get('schoolName')
            # Đọc file train từ người dùng upload
            excel_file = request.files['excel_file']
            df_train = pd.read_excel(excel_file, header=None, skiprows=1)
            excel_file_original = Path(f'Data/{school_name}/') /'TN_XH.xlsx'
            df_original = pd.read_excel(excel_file_original, header=None)

            df_combine = pd.concat([df_original, df_train], ignore_index=True)
            excel_output = Path(f'Data/{school_name}/') /'TN_XH_update.xlsx'
            df_combine.to_excel(excel_output, index=False, header=False)

            # Train các model song song
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(load_and_train, school_name,"LR_TN_XH.pkl", model_type='joblib', df = df_train, type_sujects='TN_XH')
                future2 = executor.submit(load_and_train, school_name,"MLP_TN_XH.keras", model_type='keras', df = df_train, type_sujects='TN_XH')
                future3 = executor.submit(load_and_train, school_name,"LSTM_TN_XH.keras", model_type='keras', df = df_train, type_sujects='TN_XH')

                result1 = future1.result()
                result2 = future2.result()
                result3 = future3.result()

            # Chuyển đến trang hiển thị kết quả
            return render_template(
                "PageAdminTrainDisplay.html",
                model_results=[result1, result2, result3],
                type_train = "TN_XH",
                school_name = school_name
            )

        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            return {"error": str(e)}, 500

    # Hiển thị trang upload file
    return render_template("PageAdminTrainDisplay.html")



