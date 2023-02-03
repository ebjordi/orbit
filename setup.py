from setuptools import setup, find_packages

setup(
    name='gemae-orbit',
    version='0.4.3',
    author='Jordi Eguren Brown',
    author_email='jordi.eguren.brown@gmail.com',
    description='Orbit object and functional implementations from orbital parameters',
    long_description=open('README.rst').read(),
    keywords=["orbit"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
    ],
    python_requires='>=3.7',
    license='BSD-3-Clause',
)
