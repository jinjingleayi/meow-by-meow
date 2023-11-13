'''
Authors: 
'''
from typing import Union

import librosa
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import tqdm

from . import utils


class SpecgramTransformer(TransformerMixin, BaseEstimator):
    '''Transform filepaths into a specgram.
    Original code from specgram_vectorization.ipynb (Jinjing Yi,
    Brady Ali Medina).
    '''

    def fit(self, X, y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        return self

    def transform(
        self,
        X: Union[list[str], pd.Series, np.ndarray],
        flatten: bool = False,
    ) -> np.ndarray:
        '''Transform a list-like of filepaths into an array of specgrams.

        Parameters
        ----------
        X :
            The input filepaths.

        flatten :
            If True, return flattened images.

        Returns
        -------
        X_transformed :
            The specgrams
        '''

        # Check the input is good.
        X = utils.check_filepaths_input(X)

        # Specgrams
        arrs = []
        for data_fp in tqdm.tqdm(X['filepath']):
            datalib, ratelib = librosa.load(data_fp, sr=None)
            specgram = librosa.stft(datalib)
            specmag, _ = librosa.magphase(specgram)
            mel_scale_sgram = librosa.feature.melspectrogram(
                S=specmag, sr=ratelib)
            mel_spec_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
            arrs.append(mel_spec_db)

        # Get max dimensions
        shapes0 = [_.shape[0] for _ in arrs]
        max_shape0 = np.max(shapes0)
        # This is a bit unnecessary because this dimension is the same
        # for all the data, just trying to be consistent
        # with the other dimension
        shapes1 = [_.shape[1] for _ in arrs]
        # Added one for this just to make sure all rows
        # are added 0 (just to make recover easier)
        max_shape1 = np.max(shapes1) + 1

        # Pad arrays
        spec_data = []
        for i in range(len(arrs)):
            pad0 = max_shape0 - arrs[i].shape[0]
            pad1 = max_shape1 - arrs[i].shape[1]
            spec_data.append(
                np.pad(
                    arrs[i],
                    ((0, 0), (pad0, pad1)),
                    mode='constant',
                    constant_values=-1
                )
            )

        # Transferred it to numpy array
        spec_data = np.array(spec_data)

        return spec_data


class FlattenTransformer(TransformerMixin, BaseEstimator):
    '''Flatten arrays.
    '''

    def fit(self, X: np.ndarray, y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        return self

    def transform(self, X: np.ndarray):

        X_transformed = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

        return X_transformed
