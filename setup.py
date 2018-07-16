import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lprobregressor",
    version="0.0.1",
    author="Fabian Jansen & Alexander L. De Souza",
    author_email="...",
    description="A small example package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/...?",
    packages=setuptools.find_packages(),
    classifiers=(
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
