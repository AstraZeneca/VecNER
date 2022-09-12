import os
import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

current_path = os.path.dirname(os.path.realpath(__file__))
requirement_path = current_path + '/requirements.txt'
BASE_REQUIREMENTS = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as file:
        BASE_REQUIREMENTS = file.read().splitlines()

setuptools.setup(
    name="vecner",
    version="0.0.1",
    author="AZ-AI",
    description="An NER tagger for low-resource lexicons using w2vec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["vecner/"],
    package_dir={'':'.'},
    install_requires=BASE_REQUIREMENTS
)