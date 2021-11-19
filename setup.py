import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cloudpredictionframework",
    version="0.0.1",
    author="Fruktus",
    author_email="fruktusek@gmail.com",
    description="Cloud resource usage planner utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fruktus/CloudPredictionFramework",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'keras',
        'pandas',
        'sklearn',
        'numpy'
    ]
)
