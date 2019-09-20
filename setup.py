"""Define application dependencies and entry points."""
from setuptools import setup

setup(
    name="pyimg",
    version="0.0.1",
    packages=["pyimg"],
    install_requires=["pillow", "piexif"],
    entry_points={"console_scripts": ["pyimg = pyimg.main:main"]},
)

