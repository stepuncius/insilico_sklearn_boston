import datetime
import json

import numpy as np
from flask_restful import Resource, reqparse

from data_models import Prediction
from initialize import app, api, ml_model, db

MODEL_ARGS = ('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT')

parser = reqparse.RequestParser()
for arg in MODEL_ARGS:
    parser.add_argument(arg, type=float)


class MLAPI(Resource):
    def post(self):
        vector = parser.parse_args()
        prediction = float(ml_model.predict(np.array([v for v in vector.values()]).reshape(1, -1)))
        db_log = Prediction(input_data=json.dumps(vector), prediction=prediction,
                            predicted_at=datetime.datetime.now())
        db.session.add(db_log)
        db.session.commit()
        return {'prediction': prediction}


class LogAPI(Resource):
    def get(self):
        log = Prediction.query.all()
        response = []
        for entry in log:
            response.append(
                {
                    'predicted_at': str(entry.predicted_at),
                    'prediction': entry.prediction,
                    'input_data': json.loads(entry.input_data)
                }
            )
        return response


api.add_resource(MLAPI, '/predict')
api.add_resource(LogAPI, '/show')

if __name__ == '__main__':
    db.create_all()
    app.run(host='0.0.0.0', port=5000)
