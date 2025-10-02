from setuptools import setup

setup(name='dyna_gym',
      version='0.0.1',
      packages=['dyna_gym'],
      install_requires=[
            # the following two are required to install gym==0.20.0
            'setuptools==65.5.0',
            'pyglet==1.5.27',
            'gym==0.20.0',
            'torch',
            'transformers',
            'datasets',
            'accelerate>=0.24.0',
            'scipy',
            'matplotlib',
            'networkx', # for tree plotting
      ]
)
