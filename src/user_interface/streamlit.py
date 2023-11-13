import io

import altair as alt
import joblib
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.pipeline import Pipeline
import streamlit as st

from meowlib import data_handling


st.title('Meow-by-Meow')

# Load the modeling pipeline
model_filename = './data/trained_models/k5s0.2r440.pkl'
model = joblib.load(model_filename)

# Set up the padding, getting the pad size from the number of features
freq_shape = 128
time_shape = model.n_features_in_ // freq_shape
pad_transformer = data_handling.PadTransformer()
pad_transformer.max_shape0_ = freq_shape
pad_transformer.max_shape1_ = time_shape

# Preprocessing pipeline
preprocess = Pipeline([
    ('specgram', data_handling.SpecgramTransformer()),
    ('pad', pad_transformer),
    ('flatten', data_handling.FlattenTransformer()),
])

# Read the user-provided .wav file data
wav_bytes = st.file_uploader(
    label='Upload a recording of your cat.', type='wav')

if wav_bytes is not None:

    # Load the data
    wav_file = io.BytesIO(wav_bytes.getvalue())
    rate, user_sample = wavfile.read(wav_file)
    user_sample = user_sample / np.iinfo(user_sample.dtype).max

    # Format
    dt = 1. / rate
    user_sample_duration = user_sample.size * dt
    time = np.arange(0., user_sample_duration, dt)
    user_data = pd.DataFrame({
        'time': time,
        'sample': user_sample,
    })

    # Display audio
    st.audio(wav_file)

    # Display visually
    c = alt.Chart(user_data).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='time (seconds)')),
        y=alt.Y('sample', axis=alt.Axis(title='amplitude'))
    )
    st.altair_chart(c, use_container_width=True)

    # Predict
    X = [(user_sample, rate), ]
    X_transformed = preprocess.transform(X)
    classification = model.predict(X_transformed)
    behaviors = ['isolated', 'hungry', 'being brushed']
    st.write(
        "Your cat's meow is similar to that "
        f"of a cat that is {behaviors[classification[0]]}."
    )
