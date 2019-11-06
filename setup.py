"""Define application dependencies and entry points."""
from os import path

from setuptools import find_packages, setup

cwd = path.abspath(path.dirname(__file__))


def find_version(file):
    """Read version number from source file."""
    version = {}
    try:
        with open(path.join(cwd, file), "r") as f:
            version_file = f.read()
    except Exception:
        raise RuntimeError(f"Unable to open '{file}' to get version.")
    exec(version_file, version)

    try:
        version_str = version["__version__"]
    except IndexError:
        raise RuntimeError(f"Unable to get version from '{file}'")
    return version_str


def get_long_description(readme_file):
    """Read long description from README file."""
    try:
        with open(path.join(cwd, readme_file), "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ""


setup(
    name="pyimgtool",
    version=find_version("pyimgtool/version.py"),
    author="Nick Murphy",
    author_email="comfortablynick@gmail.com",
    description="Tool to help prep images for web sharing",
    long_description=get_long_description("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/comfortablynick/pyimgtool",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pillow",
        "piexif",
        "attrs",
        "sty",
        "opencv-python",
        "numpy",
        "plotille",
    ],
    entry_points={"console_scripts": ["pyimgtool = pyimgtool.__main__:main"]},
    python_requires=">=3.6",
    project_urls={
        "Bug Reports": "https://github.com/comfortablynick/pyimgtool/issues",
        "Source": "https://github.com/comfortablynick/pyimgtool",
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ],
)
