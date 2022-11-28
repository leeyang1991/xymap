# coding='utf-8'

from setuptools import setup
long_description = open('README.md').read()
setup(
    name='xymap',
    version='0.0.3',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yang Li',
    author_email='leeyang1991@gmail.com',
    packages=['xymap'],
    url='https://github.com/leeyang1991/xymap',
    python_requires='>=3',
    install_requires=[
    'xycmap',
    'numpy',
    'Pillow',
    'gdal',
    'tqdm',
    'pandas',
    ],
)