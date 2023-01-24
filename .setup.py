from setuptools import find_packages
from setuptools import setup

setup(
    name="orbit",
    version="0.7",
    description="Orbit object and functional implementations from orbital parameters",
    url="https://github.com/ebjordi/orbit.git",
    author="Jordi Eguren Brown",
    author_email="jordi.eguren.brown@gmail.com",
    packages=find_packages(),
    include_package_data=True,
#    zip_safe=False,
)
