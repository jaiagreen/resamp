
from setuptools import setup, find_packages

setup(
    name='statistics_library',
    version='1.0',
    packages=find_packages(),
    description='A custom statistics library for chi-square analysis and other statistical functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='vishanth',
    author_email='rbvish1007@gmail.com',
    url='https://github.com/vishanth10/statistics_library.git',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='statistics chi-square analysis',
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.18',
        'scipy>=1.4',
        'matplotlib>=3.1',
        'seaborn>=0.10'
    ],
)
