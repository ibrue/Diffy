from setuptools import setup, find_packages

setup(
    name="egocodec",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "opencv-python-headless>=4.7",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    description="Diffy — the difference video codec",
)
