from setuptools import find_packages, setup

setup(
    name="rl_analysis",
    packages=find_packages(),
    version="1.0.0",
    platforms=["mac", "unix"],
    description='Python package and notebooks corresponding to "Spontaneous behavior is structured by reinforcement without exogenous reward"',
    author="wingillis, jmarkow, neurojaym",
    license="",
    install_requires=[
        "pandas",
        "opencv-python-headless",
        "numpy",
        "click",
        "toolz",
        "dask",
        "multiprocess",
        "tqdm",
        "matplotlib",
        "seaborn",
        "pyarrow",
        "colorcet",
        "toml",
        "numba",
        "joblib",
        "scipy",
        "ipywidgets",
        "tbb",  # makes numba thread-safe
        "statsmodels",
    ],
    entry_points={"console_scripts": ["rl-analysis = rl_analysis.cli:cli"]},
)
