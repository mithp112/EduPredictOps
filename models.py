from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

# Bảng tài khoản Admin
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Bảng lưu thông tin Submit Data
class SubmitData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_submits = db.Column(db.Integer, default=0)        # Tổng số lần submit
    submits_today = db.Column(db.Integer, default=0)        # Số lần submit trong hôm nay
    last_day_submit = db.Column(db.DateTime)       # Ngày cuối cùng submit



# Bảng lưu hiệu suất khi dự đoán với input của người dùng
class PerformanceSingle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # Thời điểm đo
    latency = db.Column(db.Float, nullable=False)                               # Độ trễ (ms)
    throughput = db.Column(db.Float, nullable=False)                            # Throughput (request/s)
    cpu_usage = db.Column(db.Float, nullable=False)                             # Sử dụng CPU (%)
    memory_usage = db.Column(db.Float, nullable=False)                          # Sử dụng RAM (MB)


# Bảng lưu hiệu suất khi dự đoán với file Excel
class PerformanceMulti(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # Thời điểm đo
    latency = db.Column(db.Float, nullable=False)                               # Độ trễ (ms)
    throughput = db.Column(db.Float, nullable=False)                            # Throughput (request/s)
    cpu_usage = db.Column(db.Float, nullable=False)                             # Sử dụng CPU (%)
    memory_usage = db.Column(db.Float, nullable=False)                          # Sử dụng RAM (MB)
    total_predictions = db.Column(db.Integer, nullable=False)                   # Số lượng dự đoán (số dòng trong file Excel)
