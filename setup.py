
from setuptools import setup, find_packages

setup(
    name='resamp',
    version='1.7.11',
    author='Vishanth Hari Raj Balasubramanian',
    author_email='rbvish1007@gmail.com',
    description='A custom statistics library of resampling technqiues for chi-abs, boostrapping analysis and other statistical functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vishanth10/resamp.git',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.5',
        'pandas>=1.0.5',
        'scipy>=1.5.0',
        'matplotlib>=3.2.2',
        'statsmodels>=0.11.1',
        'scikit-learn>=0.23.1',
        'seaborn>=0.10'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='resampling techniques, resample, chi-abs, power analysis, relative risk, statistics, resampling chi-abs analysis',
)






# from setuptools import setup, find_packages

# setup(
#     name='statistics_library',
#     version='1.0',
#     packages=find_packages(),
#     description='A custom statistics library for chi-square analysis and other statistical functions.',
#     long_description=open('README.md').read(),
#     long_description_content_type='text/markdown',
#     author='vishanth',
#     author_email='rbvish1007@gmail.com',
#     url='https://github.com/vishanth10/statistics_library.git',
#     classifiers=[
#         'Development Status :: 3 - Alpha',
#         'Intended Audience :: Education',
#         'License :: OSI Approved :: MIT License',
#         'Programming Language :: Python :: 3',
#         'Programming Language :: Python :: 3.7',
#         'Programming Language :: Python :: 3.8',
#         'Programming Language :: Python :: 3.9',
#     ],
#     keywords='statistics chi-square analysis',
#     install_requires=[
#         'pandas>=1.0',
#         'numpy>=1.18',
#         'scipy>=1.4',
#         'matplotlib>=3.1',
#         'seaborn>=0.10'
#     ],
# )
