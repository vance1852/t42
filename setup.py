from setuptools import setup, find_packages

setup(
    name="lunar-lander-simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "lunar-lander=lunar_lander.cli:main",
        ],
    },
)
