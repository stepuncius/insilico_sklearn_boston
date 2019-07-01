import hashlib
import pickle

import yaml
from flask import Flask
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://postgres:4VZ8f5v0PmZzBop6571JcaBhzhk3wM@db:5432/sklearn_app'

db = SQLAlchemy(app)

db.create_all()

with open('best_model_info.yaml', 'r') as info:
    model_info = yaml.safe_load(info)

with open('best_model.pckl', 'rb') as model:
    data = model.read(-1)
    md5 = hashlib.md5()
    md5.update(data)
    md5_string = md5.hexdigest()
    if md5_string != model_info['serialized_model_md5']:
        raise ValueError(f'Model has incorrect md5.{md5_string} != {model_info["serialized_model_md5"]}\n Exiting...')

with open('best_model.pckl', 'rb') as model:
    ml_model = pickle.load(model)

api = Api(app)
