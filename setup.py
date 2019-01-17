#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='beatmapml_trajectory',
    version='0.1.1',
    description='Generate Auto Mod trajectory for machine learning',
    author='Youmu Chan',
    author_email='johnmave126@gmail.com',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Games/Entertainment',
    ],
    url='https://github.com/johnmave126/beatmapml',
    install_requires=[
        'numpy',
        'bezier',
        ('slider @ git+https://github.com/llllllllll/slider.git@'
         'master#egg=slider-0.1.0')
    ]
)
