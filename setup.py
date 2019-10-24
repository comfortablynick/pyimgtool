"""Define application dependencies and entry points."""
from setuptools import setup, find_packages
from os import path
import re

cwd = path.abspath(path.dirname(__file__))


def find_version(file):
    """Read version number from source file.

    Modified from:
    """
    try:
        with open(path.join(cwd, file), "r") as f:
            version_file = f.read()
    except Exception:
        raise RuntimeError(f"Unable to open '{file}' to get version.")

    # Find version in format __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)

    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


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
    install_requires=["pillow", "piexif", "attrs"],
    entry_points={"console_scripts": ["pyimgtool = pyimgtool.__main__:main"]},
    python_requires=">=3.6",
    project_urls={
        "Bug Reports": "https://github.com/comfortablynick/pyimgtool/issues",
        "Source": "https://github.com/comfortablynick/pyimgtool",
    },
)
