import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / 'README.md').read_text()


setup(
    name='firepy',
    version='0.1.0',
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
    packages=['firepy'],
    install_requires=[
        'dill>=0.3.1.1',
        'eppy>=0.5.52',
        'esoreader>=1.2.3',
        'Flask>=1.1.1',
        'matplotlib>=3.1.3',
       ' numpy>=1.17.3',
        'olca-ipc>=0.0.8',
        'pandas>=0.25.2',
        'redis>=3.3.11',
        'requests>=2.22.0',
        'SQLAlchemy>=1.3.11',
    ],
    python_requires='>=3.7',
)
