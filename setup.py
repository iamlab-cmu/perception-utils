"""
Intelligent Autonomous Manipulation Lab Perception Utilities
Author: Jacky
"""
from setuptools import setup

requirements = [
    'numpy',
    'opencv-python',
    'apriltag',
    'autolab_core',
    'pyrealsense2',
    'autolab-perception',
]

setup(name='iamlab_perception_utils',
      version='0.0.1',
      description='Perception utilities for the Intelligent Autonomous Manipulation Lab',
      author='Jacky Liang',
      author_email='jackyliang@cmu.edu',
      package_dir = {'': '.'},
      packages=['perception_utils'],
      install_requires = requirements
)
