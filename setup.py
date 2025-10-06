from setuptools import find_packages,setup
from typing import List

HYPEN_E = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPEN_E in requirements:
            requirements.remove(HYPEN_E)

    return requirements

setup(
    name='Sentiment-Analysis-Of-Financial-News',
    version='0.0.1',
    author='Devershika',
    author_email='devershikam@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
        

)