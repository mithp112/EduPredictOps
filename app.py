# app.py
from flask import Flask
from flask_mongoengine import MongoEngine
from prometheus_flask_exporter import PrometheusMetrics
from database import PerformanceSingle, PerformanceMulti, Admin
from routes.user_routes import user_blueprint
from routes.admin_routes import admin_blueprint
from prometheus_client import Gauge

app = Flask(__name__)

# Cấu hình kết nối MongoDB Atlas
app.config['MONGODB_SETTINGS'] = {
    'db': 'mlops_score_predict',
    'host': 'mongodb+srv://minhthuanmithp112:OJFtD0ysoLUDDAJ7@cluster0.lctd7nv.mongodb.net/mlops_score_predict?retryWrites=true&w=majority'
}
metrics = PrometheusMetrics(app)



single_latency_gauge = Gauge("perf_single_latency", "Latency for single prediction", ["school"])
multi_latency_gauge = Gauge("perf_multi_latency", "Latency for multi prediction", ["school"])

# Khởi tạo MongoEngine
db = MongoEngine()
db.init_app(app)

# Đăng ký blueprint
app.register_blueprint(user_blueprint, url_prefix="/user")
app.register_blueprint(admin_blueprint, url_prefix="/admin")

# Secret key
app.secret_key = '8c3f497a6d9273c8eb5b7b67365e4a1b2b'

if __name__ == "__main__":
    app.run(host="0.0.0.0")
