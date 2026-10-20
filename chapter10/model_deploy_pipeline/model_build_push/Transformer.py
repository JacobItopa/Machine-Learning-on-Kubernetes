import joblib
import pandas as pd

DEFAULT_FEATURE_NAMES = [
    "month", "day", "day_of_week", "airline", "flight_number",
    "origin_airport", "destination_airport", "scheduled_time",
    "elapsed_time", "air_time", "distance", "ORIG_CITY",
    "ORIG_STATE", "ORIG_LATITUDE", "ORIG_LONGITUDE", "DEST_CITY",
    "DEST_STATE", "DEST_LATITUDE", "DEST_LONGITUDE",
    "scheduled_departure_hour", "scheduled_departure_minute",
    "scheduled_arrival_hour", "scheduled_arrival_minute"
]

class Transformer(object):
    """
    This class loads the ordinal encoder and applies it to incoming inference data.
    """
    def __init__(self):
        try:
            self.encoder = joblib.load('FlgithsDelayOrdinalEncoder.pkl')
        except FileNotFoundError:
            self.encoder = joblib.load('FlightsDelayOrdinalEncoder.pkl')

    def transform_input(self, X, feature_names=None, meta=None):
        """
        Seldon Core calls this function to apply data transformation before prediction.
        """
        cols = feature_names if (feature_names is not None and len(feature_names) > 0) else DEFAULT_FEATURE_NAMES
        df = pd.DataFrame(X, columns=cols)
        df_transformed = self.encoder.transform(df)
        return df_transformed.to_numpy()