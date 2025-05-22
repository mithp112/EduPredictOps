# database.py
from flask_mongoengine import MongoEngine
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = MongoEngine()

# Bảng tài khoản Admin
class Admin(db.Document):
    username = db.StringField(required=True, unique=True, max_length=150)
    password_hash = db.StringField(required=True, max_length=256)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Dữ liệu Submit tổng hợp
class SubmitData(db.Document):
    total_submits = db.IntField(default=0)
    submits_today = db.IntField(default=0)
    last_day_submit = db.DateTimeField()
    school_id = db.ReferenceField('ListSchool')


# Hiệu suất dự đoán từng người
class PerformanceSingle(db.Document):
    timestamp = db.DateTimeField(default=datetime.utcnow)
    latency = db.FloatField(required=True)
    throughput = db.FloatField(required=True)
    cpu_usage = db.FloatField(required=True)
    memory_usage = db.FloatField(required=True)
    school_id = db.ReferenceField('ListSchool')


# Hiệu suất dự đoán nhiều người từ file Excel
class PerformanceMulti(db.Document):
    timestamp = db.DateTimeField(default=datetime.utcnow)
    latency = db.FloatField(required=True)
    throughput = db.FloatField(required=True)
    cpu_usage = db.FloatField(required=True)
    memory_usage = db.FloatField(required=True)
    total_predictions = db.IntField(required=True)
    school_id = db.ReferenceField('ListSchool')


# Danh sách trường học
class ListSchool(db.Document):
    schoolName = db.StringField(required=True, unique=True, max_length=150)
    years = db.ListField(db.IntField(), required=True)
    yearsInUse = db.ListField(db.IntField(), required=True)
