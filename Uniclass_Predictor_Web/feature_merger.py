# feature_merger.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureMerger(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_be_merged):
        self.columns_to_be_merged = columns_to_be_merged

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_selected = X[self.columns_to_be_merged].astype(str).fillna('')
        merged_texts = X_selected.apply(lambda row: ' '.join(row), axis=1).tolist()
        return np.array(merged_texts)
