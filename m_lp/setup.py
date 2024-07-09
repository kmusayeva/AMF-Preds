from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="m_lp",
    version="0.1.0",
    author="Khadija Musayeva",
    author_email="khmusayeva@gmail.com",
    description="Label propagation",
    license="MIT",
    url="https://github.com/kmusayeva/AMF-preds/m-lp",
    packages=find_packages(),
    install_requires=['pandas>=2.0.0', 'matplotlib>=3.9.0', 'numpy>=2.0.0', 'scipy>=1.0.0',
                      'seaborn>=0.13.2', 'scikit-learn>=1.5.0'
                      ],
    python_requires='>=3.6',
)