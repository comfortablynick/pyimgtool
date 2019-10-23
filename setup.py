"""Define application dependencies and entry points."""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyimgtool",
    version="0.0.1",
    author="Nicholas Murphy",
    author_email="comfortablynick@gmail.com",
    description="Tool to help prep images for web sharing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/comfortablynick/pyimgtool.git",
    packages=setuptools.find_packages(),
    install_requires=["pillow", "piexif"],
    entry_points={"console_scripts": ["pyimgtool = pyimgtool.main:main"]},
    python_requires=">=3.6",
)
