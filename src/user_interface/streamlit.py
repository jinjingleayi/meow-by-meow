import io

import altair as alt
import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
import streamlit as st


st.title('Meow-by-Meow')

# Read the user-provided .wav file data
wav_bytes = st.file_uploader(label='Upload a recording of your cat.', type='wav')
if wav_bytes is not None:
    wav_file = io.BytesIO(wav_bytes.getvalue())
    rate, user_sample = wavfile.read(wav_file)
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

    # Load the model from disk
    model_filename = './data/trained_models/knn.pkl'
    model = joblib.load(model_filename)

    # Format user data to be compatible
    X = np.zeros((1, model.n_features_in_))
    X[:, :time.size] = user_sample

    # Predict
    classification = model.predict(X)
    behaviors = ['isolated', 'hungry', 'being brushed']
    st.write(
        f"Your cat's meow is similar to that of cat that is {behaviors[classification[0]]}."
    )

