"""Define application dependencies and entry points."""
from setuptools import setup, find_packages
from os import path

cwd = path.abspath(path.dirname(__file__))

with open(path.join(cwd, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyimgtool",
    version="0.0.1",
    author="Nick Murphy",
    author_email="comfortablynick@gmail.com",
    description="Tool to help prep images for web sharing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/comfortablynick/pyimgtool",
    packages=find_packages(),
    install_requires=["pillow", "piexif", "attrs"],
    entry_points={"console_scripts": ["pyimgtool = pyimgtool.__main__:main"]},
    python_requires=">=3.6",
    project_urls={
        "Bug Reports": "https://github.com/comfortablynick/pyimgtool/issues",
        "Source": "https://github.com/comfortablynick/pyimgtool",
    },
)
