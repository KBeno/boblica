import pathlib
from setuptools import setup
from boblica import __version__

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / 'README.md').read_text()


setup(
    name='boblica',
    version=__version__,
    description='Building Life Cycle Assessment and Optimisation',
    long_description=README,
    long_description_content_type='text/markdown',
    url='',
    author='Benedek Kiss',
    author_email='kiss.benedek@szt.bme.hu',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    packages=[
        'boblica.app',
        'boblica.calculation',
        'boblica.model',
        'boblica.setup',
        'boblica.tools',
    ],
    install_requires=[
        'dill>=0.3.1.1',
        'eppy>=0.5.52',
        'esoreader>=1.2.3',
        'matplotlib>=3.1.3',
        'pandas>=0.25.2',
        'requests>=2.22.0',
        'SQLAlchemy>=1.3.11'
    ],
    extras_require={
        'server': [
            'Flask==1.1.1',
            'redis==3.3.11',
            'psycopg2-binary==2.8.4'
        ],
        'openLCA': [
            'olca-ipc==0.0.8',
        ]
    },
    python_requires='>=3.7',
)
