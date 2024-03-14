import setuptools

install_deps = [
    "importlib-metadata",
    "natsort",
    "rastermap>=0.9.0",
    "tifffile",
    "numpy>=1.24.3",
    "numba>=0.57.0",
    "matplotlib",
    "scipy>=1.9.0",
    "scikit-learn",
    "cellpose",
    "scanimage-tiff-reader>=1.4.1",
]

gui_deps = [
    "qtpy",
    "pyqt6",
    "pyqt6.sip",
    "pyqtgraph",
]

io_deps = ["paramiko", "nd2", "sbxreader", "h5py", "opencv-python-headless", "xmltodict"]

nwb_deps = [
    "pynwb>=2.3.2",
]

test_deps = [
    "pytest",
    "tenacity",
    "tqdm",
    "pynwb>=2.3.2",  # this is needed as test_io contains a test with nwb
    "pytest-qt>3.3.0",
]

# check if pyqt/pyside already installed
all_deps = gui_deps + nwb_deps + test_deps + io_deps

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="suite2p-haisslab",
    author="Marius Pachitariu, Carsen Stringer - Forked by Timothé Jost-Mousseau",
    author_email="timothe.jost-mousseau@pasteur.fr",
    description="Pipeline for calcium imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JostTim/suite2p",
    packages=setuptools.find_packages(),
    version="0.14.4",
    install_requires=install_deps,
    extras_require={
        "docs": [
            "sphinx>=3.0",
            "sphinxcontrib-apidoc",
            "sphinx_rtd_theme",
            "sphinx-prompt",
            "sphinx-autodoc-typehints",
        ],
        "gui": gui_deps,
        "nwb": nwb_deps,
        "io": io_deps,
        "tests": test_deps,
        "all": all_deps,
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "suite2p = suite2p.__main__:main",
            "reg_metrics = benchmarks.registration_metrics:main",
            "tiff2scanimage = scripts.make_tiff_scanimage_compatible:main",
        ]
    },
)
