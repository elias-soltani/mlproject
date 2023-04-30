from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''Reads requirements file and returns list of requirements'''
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='mlproject',
    version='0.0.1',
    description='ML project',
    author='Elias',
    author_email='egha355@aucklanduni.ac.nz',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)