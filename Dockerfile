FROM python:3.11

WORKDIR /app

COPY . /app
COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \ 
    pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]