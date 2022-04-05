from setuptools import setup, find_packages

setup(
    name='binary_nn',
    version='1.0.0',
    url='https://github.com/cezary986/binary_nn',
    author='Cezary Maszczyk',
    author_email='cmaszczyk@polsl.com',
    description='',
    packages=find_packages(),
    install_requires=[
        'numpy==1.19.5', 
        'pandas==1.2.1',
        'tabulate==0.8.9',
        'conditions==1.0.0',
    ],
)
