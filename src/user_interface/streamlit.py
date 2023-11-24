import io

import altair as alt
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sklearn.pipeline import Pipeline
import streamlit as st
import sys
from torch import Tensor
import torchaudio

# Set up settings
settings = dict(
    # Parameters
    min_duration_seconds=4.,
    rate=8000,
    behaviors=['uncomfortable', 'hungry', 'comfortable'],

    # Define the model repository and the model filename
    repo_id='zhafen/meow-by-meow-modeling',
    model_filename='k4s0.2r440.pkl',

    # Default parameters
    default_fp='./data/raw_data/zachs_cats/pip_and_chell_wet_food.m4a',
)

sys.path.append('./meowlib/')
import data_handling

st.title('Meow-by-Meow')

# Set up the padding, getting the pad size from the number of features
freq_shape = 128
time_shape = 63
pad_transformer = data_handling.PadTransformer()
pad_transformer.max_shape0_ = freq_shape
pad_transformer.max_shape1_ = time_shape

# Preprocessing pipeline
preprocess = Pipeline([
    ('specgram', data_handling.SpecgramTransformer()),
    # TODO: Rolling Window Parameters are not in physical, intuitive units
    ('split', data_handling.RollingWindowSplitter()),
    ('pad', pad_transformer),
    ('flatten', data_handling.FlattenTransformer()),
])

# Read the user-provided .wav file data
user_file = st.file_uploader(
    label=('Upload a recording of your cat. ')
)

# Default case
if user_file is None:
    audio = AudioSegment.from_file(settings['default_fp'])
else:
    try:
        filetype = user_file.type.split('/')[-1]

        # Rename for functionality
        if filetype == 'x-m4a':
            filetype = 'm4a'

        audio = AudioSegment.from_file(user_file, format=filetype)

        if audio.duration_seconds < settings['min_duration_seconds']:
            st.error(
                'Please upload a recording longer than '
                f"{settings['min_duration_seconds']:.2g} seconds."
            )
            assert False
        else:
            st.success('File uploaded!')

    except (AssertionError, CouldntDecodeError, IndexError) as e:
        st.error('Could not load user file. Using default file.')
        audio = AudioSegment.from_file(settings['default_fp'])

st.write(audio)

with st.spinner(text="Interpreting your cat's meows..."):

    # Extract the core data
    sample = np.array(audio.get_array_of_samples())
    sample = sample / np.iinfo(sample.dtype).max

    # Resample to the right rate
    resampler = torchaudio.transforms.Resample(
        audio.frame_rate,
        settings['rate']
    )
    sample = np.array(resampler(Tensor(sample)))
    rate = settings['rate']

    dt = 1. / settings['rate']
    sample_duration = sample.size * dt

    # Preprocess
    X = [(sample, rate)]
    X_transformed = preprocess.fit_transform(X)

    # Download the model file from the Hugging Face Hub
    model_file = hf_hub_download(
        repo_id=settings['repo_id'],
        filename=settings['model_filename']
    )

    # Load the model using joblib
    model = joblib.load(model_file)

    # Predict
    classifications = model.predict(X_transformed)

    # Get the typicall classification
    c_avg = np.argmax(np.bincount(classifications))

    st.write(
        f"Overall, your cat sounds {settings['behaviors'][c_avg]}."
    )

    # Format the sample for visualization
    time = np.arange(0., sample_duration, dt)
    sample_df = pd.DataFrame({
        'time': time,
        'sample': sample,
    })

    # Add the classifications
    # TODO: The window centers aren't aligned exactly
    window_centers = np.linspace(
        0,
        sample_duration,
        classifications.size,
    )
    interp_fn = interp1d(window_centers, classifications, kind='nearest')
    sample_df['classification'] = interp_fn(sample_df['time'])

    # Visualize
    c = alt.Chart(sample_df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='time (seconds)')),
        y=alt.Y('sample', axis=alt.Axis(title='amplitude')),
        color='classification',
    )
    st.altair_chart(c, use_container_width=True)
    # TODO: The colorbar automatically added to the right side is not wanted
