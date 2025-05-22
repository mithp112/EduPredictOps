FROM python:3.12.3

WORKDIR /app

ENV TF_ENABLE_ONEDNN_OPTS=0

COPY requirements.txt ./

RUN pip install --no-cache-dir --trusted-host pypi.python.org --progress-bar off -r requirements.txt

COPY . /app

EXPOSE 5000

ENV NAME=OpentoALL

CMD ["python", "app.py"]