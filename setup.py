from setuptools import setup, find_packages

setup(
    name="diffy",
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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ian brueggeman",
    author_email="108094914+ibrue@users.noreply.github.com",
    url="https://github.com/ibrue/Diffy",
    project_urls={
        "Homepage": "https://diffy.tech",
        "Source": "https://github.com/ibrue/Diffy",
        "Issues": "https://github.com/ibrue/Diffy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Video :: Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["video", "compression", "codec", "diffy", "difference", "industrial", "ai"],
)
