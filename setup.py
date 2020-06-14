import setuptools

long_description = ""

setuptools.setup(
    name="vanidl", # Replace with your own username
    version="0.0.1",
    author="Hariharan Devarajan",
    author_email="hdevarajan@hawk.iit.edu",
    description="I/O analyzer of Scientific Deep Learning applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hariharan-devarajan/dlprofiler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
