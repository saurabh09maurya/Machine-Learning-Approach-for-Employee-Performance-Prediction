from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list = None):
        self.columns = columns  # Columns to encode
        self.encoders = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'MultiColumnLabelEncoder':
        # If no columns specified, encode all columns
        self.columns = X.columns if self.columns is None else self.columns
        for col in self.columns:
            if col in X:
                le = LabelEncoder()
                le.fit(X[col].astype(str).fillna(''))
                self.encoders[col] = le
            else:
                raise ValueError(f"Column '{col}' not found in the DataFrame")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Apply transformations using the fitted encoders
        X_transformed = X.copy()
        for col in self.columns:
            if col in self.encoders:
                X_transformed[col] = self.encoders[col].transform(X[col].astype(str).fillna(''))
            else:
                raise ValueError(f"Column '{col}' was not fitted")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        # Combine fit and transform
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Inverse transform the encoded columns back to their original values
        X_inverse_transformed = X.copy()
        for col in self.columns:
            if col in self.encoders:
                X_inverse_transformed[col] = self.encoders[col].inverse_transform(X[col])
            else:
                raise ValueError(f"Column '{col}' was not fitted")
        return X_inverse_transformed
