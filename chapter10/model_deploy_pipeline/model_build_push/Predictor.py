import joblib


class Predictor(object):
    """
    Seldon Core Predictor component for serving model inference predictions.
    """
    def __init__(self):
        self.model = joblib.load('model.pkl')

    def predict(self, data_array, feature_names=None, meta=None, **kwargs):
        """
        Generates probability predictions for flight delay classification.
        """
        return self.model.predict_proba(data_array)
