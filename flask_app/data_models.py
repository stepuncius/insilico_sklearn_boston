from sqlalchemy.dialects.postgresql import JSON

from initialize import db


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    input_data = db.Column(JSON)
    prediction = db.Column(db.Float)
    predicted_at = db.Column(db.DateTime, index=True)
