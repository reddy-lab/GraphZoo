from setuptools import setup
from setuptools import find_packages


with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name = "graphzoo",
      version = "0.0.1",
      author='Anoushka',
      maintainer='Anoushka',
      description = "A PyTorch library for hyperbolic neural networks.",
      long_description=long_description,
      long_description_content_type = "text/markdown",
      packages = find_packages(),
      url='https://github.com/AnoushkaVyas/GraphZoo.git',
      include_package_data=True,
      install_requires = [
          'numpy>=1.20.1',
          'scikit-learn>=0.24.1',
          'torch>=1.10.1',
          'torchvision>=0.8.2'
          'networkx>=2.5'
      ],
      classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
      ],
      license="MIT",
)