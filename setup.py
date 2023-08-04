import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lares",
    version="0.0.31",
    author="Karime Maamari",
    author_email="maamari@usc.edu",
    description="LARES: vaLidation, evAluation and REliability Solutions",
    license="MIT",
    keywords="evaluation, validation",
    url="http://packages.python.org/lares",
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
    install_requires=[
        "openai==0.27.8",
        "nltk==3.7",
        "torch==2.0.1",
        "transformers==4.31.0",
        "rouge==1.0.1",
        "bert_score==0.3.12",
        "tqdm==4.65.0",
        "fairlearn==0.9.0",
        "datasets==2.4.0"
    ]
)

