import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="arcticpy",
    packages=setuptools.find_packages(),
    version="0.1",
    description="AlgoRithm for Charge Transfer Inefficiency Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob Kegerreis",
    author_email="jacob.kegerreis@durham.ac.uk",
    url="https://github.com/jkeger/arcticpy",
    license="GNU GPLv3+",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    ],
    python_requires=">=3",
    install_requires=["numpy", "scipy"],
    keywords=["charge transfer inefficiency correction"],
)
