from setuptools import setup

setup(
    name="ModelQuantization",
    url="https://github.com/carminacodre/ModelQuantization",
    author="Carmina Codre",
    author_email="carminacodre@gmail.com",
    packages=['quantization'],
    install_requires=['keras', 'tensorflow>=1.7.0'],
    version='0.1',
    license='MIT',
    description='Model quantization using tensorflow',
    long_description=open('README.md').read()
)