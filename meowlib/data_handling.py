'''
'''
from typing import Union, Tuple

import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.io import wavfile
from sklearn.base import BaseEstimator, TransformerMixin
import tqdm

import utils


class WavLoader(TransformerMixin, BaseEstimator):
    '''Converts filepaths to audio sample, rate tuples.
    '''

    def fit(self, X, y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        return self

    def transform(
        self,
        X: Union[list[str], pd.Series, np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        '''The actual conversion.

        Parameters
        ----------
        X :
            The input filepaths.

        Returns:
        X_transformed :
            (data, rate) for each file.
        '''

        # Check the input is good.
        X = utils.check_filepaths_input(X)

        X_transformed = []
        for data_fp in tqdm.tqdm(X['filepath']):
            rate, data = wavfile.read(data_fp)

            # Convert to what librosa expects
            data = data / np.iinfo(data.dtype).max

            X_transformed.append((data, rate))

        return X_transformed


class FFMPEGLoader(TransformerMixin, BaseEstimator):

    def __init__(self, rate: int = 8000):
        self.rate = rate

    def fit(self, X, y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        return self

    def transform(
        self,
        X: Union[list[str], pd.Series, np.ndarray],
    ) -> Tuple[np.ndarray, int]:
        '''The actual conversion.

        Parameters
        ----------
        X :
            The input filepaths.

        Returns:
        X_transformed :
            (data, rate) for each file.
        '''

        # Check the input is good.
        X = utils.check_filepaths_input(X)

        X_transformed = []
        for data_fp in tqdm.tqdm(X['filepath']):

            # Load the data
            audio = AudioSegment.from_file(data_fp)

            # Convert to what librosa expects
            data = np.array(audio.get_array_of_samples())
            data = data / np.iinfo(data.dtype).max

            X_transformed.append((data, audio.frame_rate))

        return X_transformed


class SpecgramTransformer(TransformerMixin, BaseEstimator):
    '''Transform a list audio samples and rates into a list of specgrams.
    Original code from specgram_vectorization.ipynb (Jinjing Yi,
    Brady Ali Medina).
    '''

    def __init__(self, sample_rate: int = 8000, base_n_fft: int = 2048):

        self.sample_rate = sample_rate
        self.base_n_fft = base_n_fft

    def fit(self, X, y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        return self

    def transform(
        self,
        X: Union[list, np.ndarray],
    ) -> list[np.ndarray]:
        '''Transform a list audio samples and rates into a list of specgrams.

        Parameters
        ----------
        X :
            The input filepaths.

        Returns
        -------
        X_transformed :
            The specgrams.
        '''

        # Specgrams
        arrs = []
        for datalib, ratelib in tqdm.tqdm(X):
            n_fft = int(self.base_n_fft * ratelib / self.sample_rate)
            specgram = librosa.stft(datalib, n_fft=n_fft)
            specmag, _ = librosa.magphase(specgram)
            mel_scale_sgram = librosa.feature.melspectrogram(
                S=specmag, sr=ratelib)
            mel_spec_db = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
            arrs.append(mel_spec_db)

        return arrs


class RollingWindowSplitter(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        window_size: int = 63,
        window_spacing: int = 16,
        concatenate: bool = True,
    ):
        self.window_size = window_size
        self.window_spacing = window_spacing
        self.concatenate = concatenate

    def fit(self, X, y=None):
        '''The scikit-learn pipeline requires transformers to have a "fit"
        method, but it can be empty.
        '''

        assert self.window_size % 2 == 1, 'Window size must be odd.'
        self.window_halfsize_ = (self.window_size - 1) // 2

        return self

    def transform(
        self,
        X: Union[list, np.ndarray],
    ) -> list[np.ndarray]:
        '''Transform a list audio samples and rates into a list of specgrams.

        Parameters
        ----------
        X :
            The input filepaths.

        Returns
        -------
        X_transformed :
            The specgrams.
        '''

        # Specgrams
        split_specgrams = []
        for full_specgram in X:

            window_centers = np.arange(
                self.window_halfsize_,
                full_specgram.shape[1],
                self.window_spacing,
            )

            specgrams = [
                full_specgram[
                    :,
                    wc - self.window_halfsize_:wc + self.window_halfsize_ + 1
                ]
                for wc in window_centers
            ]
            split_specgrams.append(specgrams)

        if self.concatenate:
            if len(split_specgrams) == 1:
                split_specgrams = split_specgrams[0]
            else:
                raise NotImplementedError('Concatenate does not work yet.')

        return split_specgrams


class PadTransformer(TransformerMixin, BaseEstimator):
    '''Pads 2D arrays to the maximum dimensions of the fitted data.

    Original code from specgram_vectorization.ipynb (Jinjing Yi,
    Brady Ali Medina).
    '''

    def fit(self, X: list[np.ndarray], y=None):
        '''Get max dimensions of fitted data.
        '''

        # Get max dimensions
        shapes0 = [_.shape[0] for _ in X]
        self.max_shape0_ = np.max(shapes0)
        # This is a bit unnecessary because this dimension is the same
        # for all the data, just trying to be consistent
        # with the other dimension
        shapes1 = [_.shape[1] for _ in X]
        # Added one for this just to make sure all rows
        # are added 0 (just to make recover easier)
        self.max_shape1_ = np.max(shapes1) + 1

        return self

    def transform(self, X: list[np.ndarray]) -> np.ndarray:

        # Pad arrays
        spec_data = []
        for i in range(len(X)):
            pad0 = self.max_shape0_ - X[i].shape[0]
            pad1 = self.max_shape1_ - X[i].shape[1]
            spec_data.append(
                np.pad(
                    X[i],
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
