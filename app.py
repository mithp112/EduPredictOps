from flask import Flask
from models import db
from routes.user_routes import user_blueprint
from routes.admin_routes import admin_blueprint
from flask_migrate import Migrate



app = Flask(__name__)

# Cấu hình database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///submit_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Khởi tạo SQLAlchemy
db.init_app(app)

with app.app_context():
     db.create_all()

migrate = Migrate(app, db)

# Đăng ký blueprint
app.register_blueprint(user_blueprint, url_prefix="/user")
app.register_blueprint(admin_blueprint, url_prefix="/admin")


app.secret_key = '8c3f497a6d9273c8eb5b7b67365e4a1b2b'

if __name__ == "__main__":
    app.run(host="0.0.0.0")