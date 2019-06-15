FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy pandas matplotlib seaborn sklearn

WORKDIR /app

COPY "boston_housing.py" /app

CMD ["python3", "-u", "./boston_housing.py"]
