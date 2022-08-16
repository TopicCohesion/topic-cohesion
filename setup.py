import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "topic_cohesion",
    version="0.1.0",
    author = "Topic Cohesion",
    author_email = "topic.cohesion@gmail.com",
    description = "Cohesion measurement to evaluate partition",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/TopicCohesion/topic-cohesion",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'nltk==3.7',
        'scikit-learn==1.0.1',
        'pandas==1.1.5',
        'transformers==4.20.1',
        'tensorflow==2.9.1',
        'torch'
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = "~=3.7"
)