from setuptools import setup, find_packages

setup(
    name="wandas",
    version="0.1.0",
    description="Signal processing library in Python, inspired by pandas",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/wandas",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "librosa",
        "mosqito",
        "ipywidgets",
        # Add other dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
