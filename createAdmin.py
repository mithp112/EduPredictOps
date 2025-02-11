from flask import Flask
from models import db, Admin  # Import instance `db` từ models.py

app = Flask(__name__)

# Cấu hình đường dẫn tới cơ sở dữ liệu SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///submit_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Gắn app với instance db đã được tạo trong models.py
db.init_app(app)

with app.app_context():
    # Xóa bảng Admin (nếu tồn tại)
    Admin.__table__.drop(db.engine)

    # Tạo lại tất cả các bảng
    db.create_all()

    # Tạo tài khoản admin mới
    admin = Admin(username="admin")  # Thay đổi username tại đây nếu cần
    admin.set_password("admin")      # Thay đổi mật khẩu tại đây nếu cần

    # Lưu vào database
    db.session.add(admin)
    db.session.commit()

    print("Admin account has been created successfully!")
