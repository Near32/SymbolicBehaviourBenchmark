from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='SymbolicBehaviourBenchmark',
    version='0.0.1',
    description='Suite of OpenAI Gym-compatible multi-agent reinforcement learning environment to benchmark for behavioral traits pertaining to symbolic behaviours (primarily: being receptive, constructive, malleable, and separable) using referential games.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Near32/SymbolicBehaviourBenchmark',
    classifiers=[
      "Intended Audience :: Developers",
      "Topic :: Scientific/Engineering :: Artificial Intelligence ",
      "Programming Language :: Python :: 3",
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=[
      'numpy',
      'gym',
    ],

    python_requires=">=3.6",
)