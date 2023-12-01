import altair as alt
from huggingface_hub import hf_hub_download
import joblib
import numpy as np
import os
import pandas as pd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from scipy.interpolate import interp1d
from scipy.io import wavfile
from sklearn.pipeline import Pipeline
import streamlit as st
import sys
from tensorflow import keras
from torch import Tensor
import torchaudio

sys.path.append('./meowlib/')
import data_handling

# TODO: Do more preprocessing pre-specgram.
# E.g. cut out leading silence, pad the end, resample.
# The motivation is better consistency for specgrams.

# Preprocessing will be some combination of these
preprocessing_steps = {
    'specgram': {
        'description': 'Building specgrams...',
        'transformer': data_handling.SpecgramTransformer(),
    },
    'window_split': {
        'description': 'Splitting the recording into windows...',
        'transformer': data_handling.RollingWindowSplitter(),
    },
    'pad': {
        'description': 'Padding the data...',
        'transformer': data_handling.PadTransformer((128, 63)),
    },
    'reshape': {
        'description': 'Reshaping the data...',
        'transformer': data_handling.ReshapeTransformer(),
    }
}

# Set up settings
settings = dict(
    # Analysis parameters
    min_duration_seconds=4.,
    rate=8000,
    behaviors=['uncomfortable', 'hungry', 'comfortable'],
    n_decibel_bins=128,

    # Define the model repository and the model filename
    repo_id='zhafen/meow-by-meow',
    models={
        'CNN': {
            'filename': 'CNN_dataaug_with_2freqtime1rand_masking_bgs.keras',
            'preprocessing': ['specgram', 'window_split', 'pad', 'reshape'],
            'classifications': ['sex', 'breed', 'situation']
        },
        'CNN: situation only': {
            'filename': \
                'CNN_dataaug_with_2freqtime1rand_masking_situation.keras',
            'preprocessing': ['specgram', 'window_split', 'pad', 'reshape'],
            'classifications': ['situation']
        },
        'KNN': {
            'filename': 'situation_k3s0.2r440.pkl',
            'preprocessing': ['specgram', 'window_split', 'pad', 'reshape'],
            'classifications': ['sex', 'breed', 'situation']
        },
    },

    # Data parameters
    default_fp='./data/processed_data/combined_meows.mp3',

    # Aesthetic parameters
    color_scheme='tableau10',
)
st.set_page_config(layout="wide")
st.title('Meow-by-Meow')

# Advanced settings
st.sidebar.header('Advanced settings')
technical = st.sidebar.checkbox('Display details')
model_key = st.sidebar.selectbox(
    'Select the ML model to use',
    options=settings['models'].keys(),
    format_func=lambda fn: fn.split('_')[0],
)
model_settings = settings['models'][model_key]

# Intro
st.caption(
    'Created by Brady Ali, Jinjing Yi, Tantrik Mukerji, William Craig, '
    'and Zach Hafen-Saavedra '
    'during the Erdos Institute Fall 2023 Program.'
)
st.write(
    "Use machine learning to interpret your cat's "
    "chirps, complaints, yowls, yells, vocalizations, and meows."
)
if technical:
    st.markdown(
        ':green[Highlighted contributions:]\n'
        '  * :green[Brady Ali: spectrogram data analysis, '
        'convolutional neural network (CNN) modeling]\n'
        '  * :green[Jinjing Yi: data augmentation, spectrogram data analysis, '
        'CNN modeling]\n'
        '  * :green[Tantrik Mukerji: classification data handling, '
        'CNN modeling]\n'
        '  * :green[William Craig: K-nearest neighbors '
        'modeling]\n'
        '  * :green[Zach Hafen-Saavedra: user interface, project and code '
        'management, asset management]\n'#', project prototype]\n'
    )
    st.markdown(
        ':green[You can find the source code for meow-by-meow '
        'on [github](https://github.com/jinjinglunayi/meow-by-meow). '
        'This app is powered by [streamlit](https://streamlit.io/), '
        'a convenient tool for buidling data-science dashboards.'
        ']'
    )
st.divider()


# Read the user-provided .wav file data
st.subheader('Upload a recording of your cat')
user_file = st.file_uploader(
    label=(
        'Accepted formats include .wav, .m4a, and anything accepted by ffmpeg.'
    )
)

if technical:
    st.markdown(
        ':green[File loading is handled by pydub, '
        'which can handle a wide variety of formats.]'
    )
    st.write(
        f":green[Files cannot be shorter than {settings['min_duration_seconds']} "
        'seconds, to ensure there is sufficient data to interpret.]'
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

        # TODO: We should be able to upload shorter files, at least as short
        #       as the shortest training sample.
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
        user_file = None

st.divider()

st.subheader('The recording to interpret')

# Upload message
if user_file is None:
    st.write(
        "No recording handy? "
        "We have one ready!"
    )

if technical:
    # TODO: Either ensure the traning is working okay, or remove this.
    st.markdown(
        ':green[The default file is selected to show multiple '
        'classifications. Currently it is frankensteined from several '
        'training samples, but a longer-term solution is to use a '
        'developer-uploaded recording that captures multiple scenarios.]'
    )

st.write('Here is the audio we will interpret:')
st.write(audio)

st.divider()

st.subheader('Our analysis')

if technical:
    st.markdown(
        ':green[All our models are available online at '
        '[the Hugging Face model hub]'
        '(https://huggingface.co/zhafen/meow-by-meow). '
        'When the streamlit app runs it retrieves the requested model from '
        'Hugging Face.]'
    )

    st.markdown(
        ':green[The status box lists each of the steps in the analysis.'
        'The computations are fast enough that you may not actually see '
        'the progress indicators.]'
    )

    st.markdown(
        ':green[The ML models are trained on data of a length 4 seconds '
        'or less. To feed in user data we create a rolling window '
        'with a 4 second width, and roughly 1 second spacings '
        'between windows.]'
    )

with st.status('Interpreting...', expanded=technical):

    st.write('Retrieving the trained AI model...')
    # Download the model file from the Hugging Face Hub
    model_file = hf_hub_download(
        repo_id=settings['repo_id'],
        filename=model_settings['filename'],
    )

    # Load the model using joblib
    model_format = os.path.splitext(model_settings['filename'])[-1]
    if model_format == '.pkl':
        model = joblib.load(model_file)
    elif model_format == '.keras':
        model = keras.models.load_model(model_file)
    else:
        raise ValueError(
            f'Unrecognized model format, {model_format}. '
            "Viable options are ['.pkl', '.keras']"
        )

    st.write('Formatting data...')
    # Extract the core data
    sample = np.array(audio.get_array_of_samples())
    sample = sample / np.iinfo(sample.dtype).max

    # Resample to the right rate
    st.write('Resampling to a new frequency...')
    resampler = torchaudio.transforms.Resample(
        audio.frame_rate,
        settings['rate']
    )
    sample = np.array(resampler(Tensor(sample)))
    rate = settings['rate']

    dt = 1. / settings['rate']
    sample_duration = sample.size * dt

    # Set up preprocessing parameters based on the model
    if hasattr(model, 'inputs'):
        input_shape = model.inputs[0].shape
        assert input_shape[1] == settings['n_decibel_bins'], (
            f"Expected the number of decibel bins to be "
            f"{settings['n_decibel_bins']}. "
            f"The model has {input_shape[1]}."
        )
        time_size = input_shape[2]
        pad_shape = (settings['n_decibel_bins'], time_size)
        X_shape = (-2, settings['n_decibel_bins'], time_size, 1)
    elif hasattr(model, 'n_features_in_'):
        time_size = model.n_features_in_ // settings['n_decibel_bins']
        pad_shape = (settings['n_decibel_bins'], time_size)
        X_shape = (-2, pad_shape[0] * pad_shape[1])
    preprocessing_steps['window_split']['transformer'].set_params(
        window_size=time_size)
    preprocessing_steps['pad']['transformer'].set_params(shape=pad_shape)
    preprocessing_steps['reshape']['transformer'].set_params(shape=X_shape)

    # The preprocessing
    preprocess = Pipeline([
        (
            preprocessing_steps[key]['description'],
            preprocessing_steps[key]['transformer']
        )
        for key in model_settings['preprocessing']
    ])
    for key, item in preprocess.named_steps.items():
        st.write(key)

    # Preprocess
    X = [(sample, rate)]
    X_transformed = preprocess.fit_transform(X)

    st.write('Employing the model...')

    # Predict
    classifications = model.predict(X_transformed)

    st.write('Postprocessing...')

    # Apply map if relevant
    if len(classifications.shape) > 1:
        classifications = np.argmax(classifications, axis=1)

    behavior_mapping = {
        'isolation': 0,
        'food': 1,
        'brushing': 2,
    }
    if model_settings['classifications'] == ['sex', 'breed', 'situation']:
        label_mapping = [
            (0, 'european_shorthair', 'brushing'),
            (0, 'european_shorthair', 'food'),
            (0, 'european_shorthair', 'isolation'),
            (0, 'maine_coon', 'brushing'),
            (0, 'maine_coon', 'food'),
            (0, 'maine_coon', 'isolation'),
            (1, 'european_shorthair', 'brushing'),
            (1, 'european_shorthair', 'food'),
            (1, 'european_shorthair', 'isolation'),
            (1, 'maine_coon', 'brushing'),
            (1, 'maine_coon', 'food'),
            (1, 'maine_coon', 'isolation'),
        ]
        classifications = np.array([
            behavior_mapping[label_mapping[_][-1]]
            for _ in classifications
        ])
    elif model_settings['classifications'] == ['situation']:
        label_mapping = ['brushing', 'food', 'isolation']
        classifications = np.array([
            behavior_mapping[label_mapping[_]]
            for _ in classifications
        ])
    else:
        raise ValueError(
            f"Unclear how to handle classifications = "
            f"{model_settings['classifications']}"
        )

st.divider()

st.subheader('Our interpretation!')

if technical:
    st.markdown(
        ':green[The results are visualized using '
        '[Vega-Altair](https://altair-viz.github.io/), '
        'which is a Python API for Vega-Lite, which is built on Vega, '
        'which draws on tools such as Prefuse, Protovis, and D3.js.]'
    )

    st.markdown(
        ':green[We color the lines by the classification of the slice of '
        'data with the window center nearest to that time.]'
    )

    st.markdown(
        ':green[For short periods of time, we do not actually expect cats '
        'to actually have wildly varying situations. However, we still '
        'perform a meow-by-meow analysis to get a robust estimate.]'
    )

with st.spinner('Visualizing...'):

    # Get the typicall classification
    c_avg = np.argmax(np.bincount(classifications))

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
    sample_df['behavior'] = \
        sample_df['classification'].map(pd.Series(settings['behaviors']))

    # Visualize
    # TODO: Represent the fact that we have an averaging window somehow.
    # TODO: Does it really make sense for sentiment to change from one meow
    #       to the next? Probably not, but there can be multiple emotions
    #       so maybe some shine through more-strongly?
    c = alt.Chart(sample_df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='time (seconds)')),
        y=alt.Y(
            'sample',
            axis=alt.Axis(title='amplitude', tickCount=0),
        ),
        # TODO: This creates independent lines. We don't want that.
        color=alt.Color(
            'behavior:N',
        ).scale(scheme=settings['color_scheme']),
    )
    c = c.configure_axisX(
        # Change gridline locations
        values=window_centers,
    )
    c = c.configure_axisY(
        # Remove gridlines
        grid=False
    )
    st.altair_chart(c, use_container_width=True)

    # TODO: An animated video that plays when it records would be neat.
    # TODO: Is "amplitude" intuitive enough for a y-axis label?

    st.markdown(
        (
            "*<p style='text-align: center; font-size: larger'>"
            "*Overall**, your cat sounds "
            f"***{settings['behaviors'][c_avg]}***."
            "</p>"
        ),
        unsafe_allow_html=True,
    )
