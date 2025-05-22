import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')
from pathlib import Path




subjects_10_11_12 = [
    "Maths_1_12", "Literature_1_12", "Physics_1_12", "Chemistry_1_12", "Biology_1_12",
    "History_1_12", "Geography_1_12", "English_1_12", "Civic Education_1_12",
    "Maths_2_12", "Literature_2_12", "Physics_2_12", "Chemistry_2_12", "Biology_2_12",
    "History_2_12", "Geography_2_12", "English_2_12", "Civic Education_2_12"
]

# Khai báo các môn học kết quả cho dự đoán điển Tốt nghiệp Tự Nhiên
subjects_TN = ["Maths", "Literature", "Physics", "Chemistry", "Biology", "English"]

# Khai báo các môn học kết quả cho dự đoán điển Tốt nghiệp Xã Hội
subjects_XH = ["Maths", "Literature", "History", "Geography", "Civic Education", "English"]


# Hàm tính toán độ chính xác
def calculate_accuracy(excel_file, subjects):
    # Đọc file excel chứa kết quả dự đoán và kết quả thực tế
    df = pd.read_excel(excel_file)
    accuracy_dict = {}
    for subject in subjects:
        actual = df[subject]
        pred = df[f'{subject}_pred']
        mape = mean_absolute_percentage_error(actual, pred)
        accuracy_dict[subject] = 100 - mape * 100  # Tính độ chính xác
    return accuracy_dict



# Vẽ biểu đồ cột thể hiện độ chính xác cho mô hình
def plot_individual_model_accuracy(model_name, excel_file, subjects):
    # Tính toán độ chính xác 
    accuracies = calculate_accuracy(excel_file, subjects)
    subject_names = list(accuracies.keys())
    accuracy_values = list(accuracies.values())
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(subject_names, accuracy_values, color='green', alpha=0.7, width=0.4)
    ax.set_ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.1f}%', ha='center', va='bottom')
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} - Accuracy by Subject')
    ax.set_xticklabels(subject_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'static/Images/{model_name}_accuracy.png')
    plt.close()



# Vẽ biểu đồ so sánh độ chính xác 3 model
def plot_individual_model_accuracy_comparison(type, excel_file1, excel_file2, excel_file3, subjects):
    # Tính toán độ chính xác cho từng mô hình
    accuracies_model1 = calculate_accuracy(excel_file1, subjects)
    accuracies_model2 = calculate_accuracy(excel_file2, subjects)
    accuracies_model3 = calculate_accuracy(excel_file3, subjects)
    
    # Lấy danh sách tên môn học và giá trị độ chính xác cho từng mô hình
    subject_names = list(accuracies_model1.keys())
    accuracy_values_model1 = list(accuracies_model1.values())
    accuracy_values_model2 = list(accuracies_model2.values())
    accuracy_values_model3 = list(accuracies_model3.values())

    # Số lượng môn học
    n_subjects = len(subject_names)
    
    # Thiết lập vị trí cho các nhóm cột
    x = np.arange(n_subjects)
    
    # Chiều rộng của từng cột
    width = 0.2

    # Tạo figure và axes cho biểu đồ
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Vẽ các cột cho từng mô hình
    bars1 = ax.bar(x - width, accuracy_values_model1, width, label='Linear Regression', color='green', alpha=0.7)
    bars2 = ax.bar(x, accuracy_values_model2, width, label='Multilayer Perceptron', color='blue', alpha=0.7)
    bars3 = ax.bar(x + width, accuracy_values_model3, width, label='Long Short Term Memory', color='orange', alpha=0.7)
    
    # Đặt giới hạn trục y từ 0 đến 100
    ax.set_ylim(0, 100)
    
    # Thêm giá trị lên đầu các cột
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
    
    # Đặt tiêu đề và nhãn trục
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Subject for 3 Models')
    
    # Đặt tên cho trục x với danh sách môn học và điều chỉnh góc độ
    ax.set_xticks(x)
    ax.set_xticklabels(subject_names, rotation=45, ha='right')
    
    # Thêm chú thích cho các mô hình
    ax.legend()
    
    # Điều chỉnh layout để tránh bị cắt nhãn
    plt.tight_layout()
    
    # Lưu biểu đồ dưới dạng file ảnh
    plt.savefig(f'static/Images/{type}_accuracy_comparison.png')
    plt.close()


# Vẽ biểu đồ cột thể hiện phổ điểm cho mô hình
def plot_subjects_point_spectrum(model_name, file_path, subjects):

    df = pd.read_excel(file_path)

    # Khoảng điểm của phổ điểm là 0.5 trải từ 0 - 10 
    bins = np.arange(0, 10.5, 0.5)


    for subject in subjects:
        actual = df[subject]
        predicted = df[f'{subject}_pred']

        actual_hist, _ = np.histogram(actual, bins=bins)
        predicted_hist, _ = np.histogram(predicted, bins=bins)


        plt.figure(figsize=(10, 6))
        plt.plot(bins[:-1], actual_hist, label=f'Actual {subject}', marker='o')
        plt.plot(bins[:-1], predicted_hist, label=f'Predicted {subject}', marker='o', linestyle='--')
        plt.xlabel('Score Ranges')
        plt.ylabel('Number of Scores')
        plt.title(f'Distribution of Scores for {subject}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'static/Images/{model_name}_{subject}_spectrum.png')
        plt.close()



# Hàm tính điểm trung bình môn học
def calculate_subject_average(hk1, hk2):
    return round((hk1 + hk2 * 2) / 3, 1)

# Hàm đánh giá học sinh
def evaluate_student(averages):
    try:
        if averages['GPA'] >= 8.0 and all(averages[subject] >= 6.5 for subject in averages if subject != 'GPA') and (averages['Toán'] >= 8.0 or averages['Văn'] >= 8.0):
            return 'Giỏi'
        elif averages['GPA'] >= 6.5 and all(averages[subject] >= 5.0 for subject in averages if subject != 'GPA') and (averages['Toán'] >= 6.5 or averages['Văn'] >= 6.5):
            return 'Khá'
        elif averages['GPA'] >= 5.0 and all(averages[subject] >= 3.5 for subject in averages if subject != 'GPA') and (averages['Toán'] >= 5.0 or averages['Văn'] >= 5.0):
            return 'Trung bình'
        else:
            return 'Yếu'
    except KeyError as e:
        print(f"Lỗi KeyError: {e} - Có thể tên môn học bị sai hoặc không tồn tại trong dữ liệu.")
        raise

# Hàm xử lý từng học sinh
def count_evaluate_students(type, excel_file):
    df = pd.read_excel(excel_file, header=None)
    
    if type == '10_11_12':
        n = 3
        grade_counts_by_year = {
            'Lớp 10': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0},
            'Lớp 11': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0},
            'Cả 2 năm': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0}}
    else:
        n = 4
        grade_counts_by_year = {
            'Lớp 10': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0},
            'Lớp 11': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0},
            'Lớp 12': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0},
            'Cả 3 năm': {'Giỏi': 0, 'Khá': 0, 'Trung bình': 0, 'Yếu': 0}}
        

    for index, row in df.iterrows():
        student_name = row[0]
        student_result = {'Tên': student_name}
        all_year_grades = []

        for year in range(1, n):
            start_col = (year - 1) * 18 + 1
            end_col = start_col + 18
            year_columns = row[start_col:end_col]

            # Tính điểm trung bình các môn học
            subject_averages = {
                'Toán': calculate_subject_average(year_columns.iloc[0], year_columns.iloc[9]),
                'Văn': calculate_subject_average(year_columns.iloc[1], year_columns.iloc[10]),
                'Lý': calculate_subject_average(year_columns.iloc[2], year_columns.iloc[11]),
                'Hóa': calculate_subject_average(year_columns.iloc[3], year_columns.iloc[12]),
                'Sinh': calculate_subject_average(year_columns.iloc[4], year_columns.iloc[13]),
                'Sử': calculate_subject_average(year_columns.iloc[5], year_columns.iloc[14]),
                'Địa': calculate_subject_average(year_columns.iloc[6], year_columns.iloc[15]),
                'GDCD': calculate_subject_average(year_columns.iloc[7], year_columns.iloc[16]),
                'Anh': calculate_subject_average(year_columns.iloc[8], year_columns.iloc[17])
            }

            # Tính GPA (điểm trung bình toàn bộ các môn học)
            subject_averages['GPA'] = pd.Series(subject_averages).mean()

            # Đánh giá học sinh (Giỏi, Khá, Trung bình, Yếu)
            grade = evaluate_student(subject_averages)
            all_year_grades.append(grade)

            # Cập nhật loại cho từng năm
            if year == 1:
                grade_counts_by_year['Lớp 10'][grade] += 1
            elif year == 2:
                grade_counts_by_year['Lớp 11'][grade] += 1
            elif year == 3 and type != '10_11_12':
                grade_counts_by_year['Lớp 12'][grade] += 1
            # Đếm loại xếp loại cho tổng 3 năm hoặc 2 năm 
            if type == '10_11_12':
                grade_counts_by_year['Cả 2 năm'][grade] += 1
            else:
                grade_counts_by_year['Cả 3 năm'][grade] += 1
        
    return grade_counts_by_year


# Vẽ biểu đồ tròn phân bố học lực
def plot_grade_distribution_pie(type, excel_file):
    # Thiết lập các nhãn và màu sắc cho các loại xếp loại
    labels = ['Giỏi', 'Khá', 'Trung bình', 'Yếu']
    colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff6666']  # Màu cho các loại xếp loại

    # Lấy dữ liệu xếp loại học sinh theo năm từ hàm `count_evaluate_students`
    grade_counts_by_year = count_evaluate_students(type, excel_file)
    
    # Tùy chỉnh số lượng năm dựa trên loại (TN hoặc XH)
    if type == '10_11_12':
        years = ['Lớp 10', 'Lớp 11', 'Cả 2 năm']  # Cho loại Tốt Nghiệp (3 phần)
        fig = plt.figure(figsize=(18, 12))
        spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 0])]
    else:
        years = ['Lớp 10', 'Lớp 11', 'Lớp 12', 'Cả 3 năm']  # Cho loại khác (4 phần)
        fig = plt.figure(figsize=(24, 12))
        spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 0]), fig.add_subplot(spec[1, 1])]
    
    # Vẽ biểu đồ tròn cho từng năm
    for i, year in enumerate(years):
        ax = axs[i]
        grade_counts = grade_counts_by_year[year]
        sizes = [grade_counts[label] for label in labels]
        ax.pie(sizes, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10})
        ax.set_title(f'Phân bố điểm xếp loại học sinh {year}', fontsize=14)
    

    
    plt.legend(labels, title="Khoảng điểm", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)) 
    plt.tight_layout()
    plt.savefig(f'static/Images/{type}_evaluate_distribution.png')
    plt.close()



def calculate_subject_average1(hk1, hk2):
    return round((hk1 + hk2 ) / 2, 1)

# Hàm xử lý và đếm số lượng học sinh theo khoảng điểm cho từng năm
def count_score_ranges(type, excel_file):
    df = pd.read_excel(excel_file, header=None)
    
    # Xác định số năm học dựa trên loại
    if type == '10_11_12':
        n = 3
        years = ['Lớp 10', 'Lớp 11', 'Cả 2 năm']
    else:
        n = 4
        years = ['Lớp 10', 'Lớp 11', 'Lớp 12', 'Cả 3 năm']
    
    # Khởi tạo dictionary để lưu số lượng học sinh trong mỗi khoảng điểm cho từng năm
    score_counts_by_year = {year: {'0-4.9': 0, '5-6.4': 0, '6.5-7.9': 0, '8-10': 0} for year in years}
    
    # Định nghĩa các khoảng điểm
    bins = [0, 4.9, 6.4, 7.9, 10]
    labels = ['0-4.9', '5-6.4', '6.5-7.9', '8-10']
    
    for index, row in df.iterrows():
        for year in range(1, n):
            start_col = (year - 1) * 18 + 1
            end_col = start_col + 18
            year_columns = row[start_col:end_col]
            
            # Tính điểm trung bình các môn học
            subject_averages = {
                'Toán': calculate_subject_average1(year_columns.iloc[0], year_columns.iloc[9]),
                'Văn': calculate_subject_average1(year_columns.iloc[1], year_columns.iloc[10]),
                'Lý': calculate_subject_average1(year_columns.iloc[2], year_columns.iloc[11]),
                'Hóa': calculate_subject_average1(year_columns.iloc[3], year_columns.iloc[12]),
                'Sinh': calculate_subject_average1(year_columns.iloc[4], year_columns.iloc[13]),
                'Sử': calculate_subject_average1(year_columns.iloc[5], year_columns.iloc[14]),
                'Địa': calculate_subject_average1(year_columns.iloc[6], year_columns.iloc[15]),
                'GDCD': calculate_subject_average1(year_columns.iloc[7], year_columns.iloc[16]),
                'Anh': calculate_subject_average1(year_columns.iloc[8], year_columns.iloc[17])
            }
            
            # Tính GPA (điểm trung bình toàn bộ các môn học)
            average_score = pd.Series(subject_averages).mean()
            
            # Phân loại vào các khoảng điểm
            score_range = pd.cut([average_score], bins=bins, labels=labels, right=False)[0]
            
            # Cập nhật số lượng học sinh trong khoảng điểm tương ứng
            if year == 1:
                current_year = 'Lớp 10'
            elif year == 2:
                current_year = 'Lớp 11'
            elif year == 3 and type != '10_11_12':
                current_year = 'Lớp 12'
            else:
                continue  # Nếu type == '10_11_12' và year >=3 thì bỏ qua

            score_counts_by_year[current_year][score_range] += 1
            
            # Cập nhật cho tổng hợp cả năm
            if type == '10_11_12' and year <= 2:
                aggregate_year = 'Cả 2 năm'
            elif type != '10_11_12' and year <=3:
                aggregate_year = 'Cả 3 năm'
            else:
                continue
            score_counts_by_year[aggregate_year][score_range] += 1
    
    return score_counts_by_year, labels

# Vẽ biểu đồ tròn phân bố điểm trung bình
def plot_score_distribution_pie(type, excel_file):
    # Đếm số lượng học sinh theo khoảng điểm cho từng năm
    score_counts_by_year, labels = count_score_ranges(type, excel_file)
    
    # Thiết lập các nhãn và màu sắc cho các khoảng điểm
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']  # Màu cho các khoảng điểm
    
    # Tùy chỉnh số lượng năm dựa trên loại (TN hoặc XH)
    if type == '10_11_12':
        years = ['Lớp 10', 'Lớp 11', 'Cả 2 năm']  # Cho loại Tốt Nghiệp (3 phần)
        fig = plt.figure(figsize=(18, 12))
        spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 0])]
    else:
        years = ['Lớp 10', 'Lớp 11', 'Lớp 12', 'Cả 3 năm']  # Cho loại khác (4 phần)
        fig = plt.figure(figsize=(24, 12))
        spec = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        axs = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 0]), fig.add_subplot(spec[1, 1])]
    
    # Vẽ biểu đồ tròn cho từng năm
    for i, year in enumerate(years):
        ax = axs[i]
        grade_counts = score_counts_by_year[year]
        sizes = [grade_counts[label] for label in labels]
        ax.pie(sizes, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10})
        ax.set_title(f'Phân bố điểm trung bình {year}', fontsize=14)
    

    plt.legend(labels, title="Khoảng điểm", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig(f'static/Images/{type}_score_distribution.png')
    plt.close()




# Tính trung bình các môn trong 6 học kỳ giảm kích thước ma trận tương quan
def calculate_averages(type, excel_file):
    # Đọc dữ liệu từ file Excel, bỏ cột đầu tiên là tên
    df = pd.read_excel(excel_file, header=None)
    if type == 'TN_TN':
        # Indices của các môn 
        subject_indices = {
            'Maths': [1, 10, 19, 28, 37, 46],
            'Literature': [2, 11, 20, 29, 38, 47],
            'Physics': [3, 12, 21, 30, 39, 48],
            'Chemistry': [4, 13, 22, 31, 40, 49],
            'Biology': [5, 14, 23, 32, 41, 50],
            'History': [6, 15, 24, 33, 42, 51],
            'Geography': [7, 16, 25, 34, 43, 52],
            'English': [8, 17, 26, 35, 44, 53],
            'Civic Education': [9, 18, 27, 36, 45, 54],
            'Math_TN': [55], 'Literature_TN': [56],
            'Physics_TN': [57], 'Chemistry_TN': [58], 'Biology': [59],
            'English_TN': [60]}
    elif type == 'TN_XH':
        subject_indices = {
            'Maths': [1, 10, 19, 28, 37, 46],
            'Literature': [2, 11, 20, 29, 38, 47],
            'Physics': [3, 12, 21, 30, 39, 48],
            'Chemistry': [4, 13, 22, 31, 40, 49],
            'Biology': [5, 14, 23, 32, 41, 50],
            'History': [6, 15, 24, 33, 42, 51],
            'Geography': [7, 16, 25, 34, 43, 52],
            'English': [8, 17, 26, 35, 44, 53],
            'Civic Education': [9, 18, 27, 36, 45, 54],
            'Math_TN': [55], 'Literature_TN': [56],
            'History_TN': [57], 'Geography_TN': [58], 'Civic Education': [59],
            'English_TN': [60]}
    else:
        subject_indices = {
            'Maths': [1, 10, 19, 28, 37, 46],
            'Literature': [2, 11, 20, 29, 38, 47],
            'Physics': [3, 12, 21, 30, 39, 48],
            'Chemistry': [4, 13, 22, 31, 40, 49],
            'Biology': [5, 14, 23, 32, 41, 50],
            'History': [6, 15, 24, 33, 42, 51],
            'Geography': [7, 16, 25, 34, 43, 52],
            'English': [8, 17, 26, 35, 44, 53],
            'Civic Education': [9, 18, 27, 36, 45, 54]}
    

    # Tạo một DataFrame mới chứa giá trị trung bình
    avg_scores = pd.DataFrame()
    # Tính trung bình của từng môn
    for subject, indices in subject_indices.items():

        avg_scores[subject] = df.iloc[:, indices].mean(axis=1)

    return avg_scores
    
# Vẽ ma trận tương quan
def plot_calculate_correlations(type, excel_file):
    avg_scores = calculate_averages(type, excel_file)
    correlation_matrix = avg_scores.corr()
    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(f'static/Images/{type}_correlations.png')
    plt.close()





# Vẽ biểu đồ thể hiện muốn quan hệ giữa dữ liệu đầu vào và đầu ra
def plot_relationship_input_output(type, file_path):
    # Đọc dữ liệu từ file Excel
    df = pd.read_excel(file_path, header=None)

    # Bỏ cột đầu tiên(Tên) và cột cuối cùng(Hoàn cảnh)
    df = df.iloc[:, 1:-1]

    if type == '10_11_12':
        # Phân chia dữ liệu
        X = df.iloc[:, :36]  # 36 cột đầu (4 học kỳ đầu)
        Y = df.iloc[:, 36]  # 18 cột cuối (2 học kỳ cuối)

        # Danh sách các môn học
        subjects = ['Maths', 'Literature', 'Physics', 'Chemistry', 'Biology', 'History', 'Geography', 'English', 'Civic Education']

        # Vẽ biểu đồ mối quan hệ giữa các môn học
        for i, subject in enumerate(subjects):
            plt.figure(figsize=(10, 6))
            plt.scatter(X.iloc[:, i], Y.iloc[:, i])
            plt.plot([0, 10], [0, 10], 'r--', label='y = x') 
            plt.xlabel(f'{subject} (4 Học kỳ đầu)')
            plt.ylabel(f'{subject} (2 Học kỳ cuối)')
            plt.title(f'Mối quan hệ giữa {subject} 4 học kỳ đầu và {subject} 2 học kỳ cuối')
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.grid(True)
            plt.savefig(f'static/Images/{type}_{subject}_relationship_input_output.png')
            plt.close()
    elif type in ['TN_TN', 'TN_XH']:
        if type == 'TN_TN':
            subjects = ['Maths', 'Literature', 'Physics', 'Chemistry', 'Biology', 'English']
        elif type == 'TN_XH':
            subjects = ['Maths', 'Literature', 'History', 'Geography', 'Civic Education', 'English']

        # Xác định các chỉ số cột cho môn học
        subject_indices = {
            'Maths': 0, 'Literature': 1, 'Physics': 2, 'Chemistry': 3, 'Biology': 4,
            'History': 5, 'Geography': 6, 'English': 7, 'Civic Education': 8
        }
        
        # Chọn cột tương ứng với môn học
        X = df.iloc[:, [subject_indices[subj] for subj in subjects]]  # Các cột của môn học
        Y = df.iloc[:, -1]  # Cột cuối cùng là điểm thi tốt nghiệp

        # Vẽ biểu đồ mối quan hệ giữa các môn học và điểm thi tốt nghiệp
        for i, subject in enumerate(subjects):
            plt.figure(figsize=(10, 6))
            plt.scatter(X.iloc[:, i], Y)
            plt.plot([0, 10], [0, 10], 'r--', label='y = x') 
            plt.xlabel(f'{subject} (6 Học kỳ)')
            plt.ylabel('Điểm thi tốt nghiệp')
            plt.title(f'Mối quan hệ giữa {subject} và điểm thi {subject} tốt nghiệp')
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.grid(True)
            plt.savefig(f'static/Images/{type}_{subject}_relationship_input_output.png')
            plt.close()

    else:
        raise ValueError("Loại dữ liệu không hợp lệ. Vui lòng chọn '10_11_12', 'TN_TN' hoặc 'TN_XH'.")









def update_chart_10_11_12(school_name):
    base_path = Path("Data") / school_name
    file_prefix = '10_11_12'

    plot_subjects_point_spectrum('LR', base_path / f'LR_Actual_Pred_{file_prefix}.xlsx', subjects_10_11_12)
    plot_subjects_point_spectrum('MLP', base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx', subjects_10_11_12)
    plot_subjects_point_spectrum('LSTM', base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx', subjects_10_11_12)

    plot_individual_model_accuracy('LR', base_path / f'LR_Actual_Pred_{file_prefix}.xlsx', subjects_10_11_12)
    plot_individual_model_accuracy('MLP', base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx', subjects_10_11_12)
    plot_individual_model_accuracy('LSTM', base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx', subjects_10_11_12)

    plot_individual_model_accuracy_comparison(
        file_prefix,
        base_path / f'LR_Actual_Pred_{file_prefix}.xlsx',
        base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx',
        base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx',
        subjects_10_11_12
    )

    plot_grade_distribution_pie(school_name, base_path / f'{file_prefix}.xlsx')
    plot_score_distribution_pie(school_name, base_path / f'{file_prefix}.xlsx')
    plot_calculate_correlations(school_name, base_path / f'{file_prefix}.xlsx')


def update_chart_TN_TN(school_name):
    base_path = Path("Data") / school_name
    file_prefix = 'TN_TN'

    plot_subjects_point_spectrum('LR_TN', base_path / f'LR_Actual_Pred_{file_prefix}.xlsx', subjects_TN)
    plot_subjects_point_spectrum('MLP_TN', base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx', subjects_TN)
    plot_subjects_point_spectrum('LSTM_TN', base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx', subjects_TN)

    plot_individual_model_accuracy('LR_TN', base_path / f'LR_Actual_Pred_{file_prefix}.xlsx', subjects_TN)
    plot_individual_model_accuracy('MLP_TN', base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx', subjects_TN)
    plot_individual_model_accuracy('LSTM_TN', base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx', subjects_TN)

    plot_individual_model_accuracy_comparison(
        file_prefix,
        base_path / f'LR_Actual_Pred_{file_prefix}.xlsx',
        base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx',
        base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx',
        subjects_TN
    )

    plot_grade_distribution_pie(school_name, base_path / f'{file_prefix}.xlsx')
    plot_score_distribution_pie(school_name, base_path / f'{file_prefix}.xlsx')
    plot_calculate_correlations(school_name, base_path / f'{file_prefix}.xlsx')


def update_chart_TN_XH(school_name):
    base_path = Path("Data") / school_name
    file_prefix = 'TN_XH'

    plot_subjects_point_spectrum('LR_XH', base_path / f'LR_Actual_Pred_{file_prefix}.xlsx', subjects_XH)
    plot_subjects_point_spectrum('MLP_XH', base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx', subjects_XH)
    plot_subjects_point_spectrum('LSTM_XH', base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx', subjects_XH)

    plot_individual_model_accuracy('LR_XH', base_path / f'LR_Actual_Pred_{file_prefix}.xlsx', subjects_XH)
    plot_individual_model_accuracy('MLP_XH', base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx', subjects_XH)
    plot_individual_model_accuracy('LSTM_XH', base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx', subjects_XH)

    plot_individual_model_accuracy_comparison(
        file_prefix,
        base_path / f'LR_Actual_Pred_{file_prefix}.xlsx',
        base_path / f'MLP_Actual_Pred_{file_prefix}.xlsx',
        base_path / f'LSTM_Actual_Pred_{file_prefix}.xlsx',
        subjects_XH
    )