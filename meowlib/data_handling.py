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
    '''

    def fit(self, X: Union[list[str], pd.Series, np.ndarray], y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        X = utils.check_filepaths_input(X)

        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: Union[list[str], pd.Series, np.ndarray],
    ) -> list[np.ndarray]:
        '''Transform a list-like of filepaths into an array of specgrams.

        Parameters
        ----------
        X :
            The input filepaths.

        Returns
        -------
        X_transformed :
            A list of specgrams.
        '''

        # Check is fit had been called
        check_is_fitted(self, 'is_fitted_')

        # Check the input is good.
        X = utils.check_filepaths_input(X)

        arrs = []
        for data_fp in tqdm.tqdm(X['filepath']):
            datalib, ratelib = librosa.load(data_fp, sr=None)
            specgram = librosa.stft(datalib)
            specmag, _ = librosa.magphase(specgram)
            mel_scale_sgram = librosa.feature.melspectrogram(
                S=specmag, sr=ratelib)
            mel_spec_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
            arrs.append(mel_spec_db)

        return arrs
