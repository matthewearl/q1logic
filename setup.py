#!/usr/bin/env python

from distutils.core import setup


setup(name='q1logic',
      version='1.0',
      entry_points={
          'console_scripts': [
              'q1logic_create_map = q1logic.map:create_map_entrypoint'
          ]
      },
      description='Q1 logic',
      install_requires=['numpy'],
      author='Matt Earl',
      packages=['q1logic'])

