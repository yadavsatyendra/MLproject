from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirments(file_path:str)->List[str]:
  """
    This function returns a list of requirements.
    """
  requirments=[]
  with open(file_path) as file:
    requirments=file.readlines()
    requirments=[req.replace("\n"," ") for req in requirments]
    if HYPEN_E_DOT in requirments:
      requirments.remove(HYPEN_E_DOT)
  return requirments

setup(
    name='MLproject',
    version='0.001',
    author='satyendra',
    author_email='nitiansatyenrdrayadav001@gmail.com',
    packages=find_packages(),
    install_requies= get_requirments('requirments.txt')
)