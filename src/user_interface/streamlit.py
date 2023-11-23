import io

import altair as alt
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.pipeline import Pipeline
import streamlit as st
import sys

sys.path.append('./meowlib/')
import data_handling

st.title('Meow-by-Meow')

# Define the model repository and the model filename
repo_id = "zhafen/meow-by-meow-modeling"
filename = "k4s0.2r440.pkl"

# Download the model file from the Hugging Face Hub
model_file = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the model using joblib
model = joblib.load(model_file)

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

# Fall back to default file
if wav_bytes is None:
    from pydub import AudioSegment

    # Load the m4a file
    default_filepath = './data/raw_data/zachs_cats/pip_and_chell_wet_food.m4a'
    audio = AudioSegment.from_file(default_filepath, format="m4a")

    # DEBUG
    st.write(audio.__dir__())
    st.write(audio.normalize.__doc__)

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

# DEBUG
# import skops.hub_utils as hub_utils
# res = hub_utils.get_model_output("zhafen/meow-by-meow-modeling", X_transformed)
# st.write(
#     "Your cat's meow is similar to that "
#     f"of a cat that is {behaviors[res[0]]}."
# )