
from setuptools import setup, find_packages

setup(
    name='statistics_library',
    version='0.1',
    packages=find_packages(),
    description='A custom statistics library for chi-square analysis and other statistical functions.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='vishanth',
    author_email='rbvish1007@gmail.com',
    url='https://github.com/vishanth10/Statistics_for_Lifesciences.git',
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
)
