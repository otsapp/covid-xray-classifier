FROM python:3.8-slim-buster

WORKDIR /app
ADD ./*requirements.txt /app/

COPY requirements.txt /
RUN pip install -r /requirements.txt

ADD . /app

CMD PYTHONPATH=src python -m xrayclassifier.trainer.training