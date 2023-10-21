from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

with open("gbt/version.py") as f:
    version_text = f.read()
    __version__ = version_text.split('"')[1]

setup(
    name="gbt",
    version=__version__,
    description="A gradient boosted tree library with automatic feature engineering.",
    url="https://github.com/zzsi/gbt",
    author="Zhangzhang Si",
    author_email="zhangzhang.si@gmail.com",
    license="MIT",
    packages=["gbt"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
