import os
from setuptools import setup, find_packages

# Read long description from README.md with fallback
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "PyTurbo - High-performance Python library for blazing-fast data analysis"

setup(
    name="pyturbo-analytics",
    version="0.1.2",  # Updated version number
    use_scm_version=True,
    packages=find_packages(),
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "dask>=2021.6.0",
        "numba>=0.53.0",
        "plotly>=5.1.0",
        "pytest>=6.2.5",
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
            'isort',
            'mypy',
            'flake8',
            'pytest-cov',  # for coverage reports
            'sphinx',      # for documentation
            'twine',       # for PyPI uploads
        ],
        'gpu': [
            'cupy-cuda11x>=10.0.0',
            'cudf-cuda11x>=22.12.0',
        ],
        'viz': [
            'plotly>=5.1.0',
            'datashader>=0.13.0',
        ],
        'all': [
            'cupy-cuda11x>=10.0.0',
            'cudf-cuda11x>=22.12.0',
            'plotly>=5.1.0',
            'datashader>=0.13.0',
        ]
    },
    entry_points={
        "console_scripts": [
            "pyturbo=pyturbo.cli:main",
            "pyturbo-profile=pyturbo.cli:profile",
            "pyturbo-benchmark=pyturbo.cli:benchmark",
        ],
    },
    author="ghassenTn",
    author_email="ghassen.xr@gmail.com",
    description="A high-performance Python library for blazing-fast data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyturbo/pyturbo",
    project_urls={
        "Bug Tracker": "https://github.com/pyturbo/pyturbo/issues",
        "Documentation": "https://pyturbo.readthedocs.io/",
        "Source Code": "https://github.com/pyturbo/pyturbo",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="data analysis, performance optimization, GPU acceleration, parallel processing, "
             "pandas, numpy, data science, high performance computing, machine learning",
    python_requires=">=3.9,<3.14",
    include_package_data=True,
    package_data={
        "pyturbo": [
            "data/*.csv",
            "templates/*.html",
            "config/*.yaml",
        ],
    },
    zip_safe=False,  # Ensure the package can be properly installed and imported
)
