FROM python:3.7-stretch

RUN mkdir /flask_app

ADD ./flask_app /flask_app
ADD ./ml_model/best_model.pckl /flask_app
ADD ./ml_model/best_model_info.yaml /flask_app

ADD ./requirements.txt /flask_app/requirements.txt

RUN pip install -r /flask_app/requirements.txt

CMD python3 /flask_app/app.py

