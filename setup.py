#!/usr/bin/env python3
"""
Setup script for ProtFunc package.

This allows the models and scripts modules to be properly imported
regardless of where you run Python from.

Installation:
    pip install -e .

This will install the package in "editable" mode, so changes to the
source code are immediately reflected without reinstalling.
"""

from setuptools import setup, find_packages

setup(
    name="protfunc",
    version="2.0.0",
    description="Protein Function Prediction with Enhanced ResidualMLP",
    author="Siddhant Bhat",
    packages=find_packages(include=["models", "models.*", "scripts", "scripts.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "aiohttp",
        "fair-esm",
        "joblib",
        "tensorboard",
    ],
    extras_require={
        "server": [
            "fastapi",
            "uvicorn",
            "pydantic",
            "python-multipart",
            "huggingface_hub",
        ],
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "protfunc-train=scripts.train_model:main",
            "protfunc-scrape=scripts.uniprot_scraper:main",
        ],
    },
)
