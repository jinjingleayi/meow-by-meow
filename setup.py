'''Installation script for meow-by-meow.
Run with `pip install -e .`
'''

import setuptools

setuptools.setup(
    name="meow-by-meow",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pytest',
	'scikit-learn',
        'jupyterlab',
        'jupyter_contrib_nbextensions',
	'librosa',
    ],
)
