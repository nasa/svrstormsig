from setuptools import setup, find_packages
import os
import shutil
from setuptools.command.develop import develop
from setuptools.command.install import install
import requests
import zipfile
#NAME     = "severe_storm_ml_detection"
#DESC     = "Package for running machine learning code on various satellite data sets"
#AUTHOR   = "John W. Cooney"
#EMAIL    = "john.w.cooney@nasa.gov"

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        working_dir = os.getcwd()
        r = requests.get('https://science-data.larc.nasa.gov/LaRC-SD-Publications/2023-05-05-001-JWC/data/ML_data.zip', allow_redirects=True)
        open('ML_data.zip', 'wb').write(r.content)
        with zipfile.ZipFile('ML_data.zip', 'r') as zip_ref:
            zip_ref.extractall(working_dir)
        
        shutil.move(os.path.join(working_dir, 'ML_data', 'data'), os.path.join(os.path.dirname(working_dir), 'data'))
        shutil.rmtree(os.path.join(working_dir, 'ML_data'))
        os.remove(os.path.join(working_dir, 'ML_data.zip'))
        os.makedirs(os.path.join(working_dir, 'glmtools_docs'), exist_ok = True)
        shutil.move(os.path.join(working_dir, 'docs'), os.path.join(working_dir, 'glmtools_docs', 'docs'))
        shutil.move(os.path.join(working_dir, 'examples'), os.path.join(working_dir, 'glmtools_docs', 'examples'))
        shutil.move(os.path.join(working_dir, 'environment.yml'), os.path.join(working_dir, 'glmtools_docs', 'environment.yml'))
        shutil.move(os.path.join(working_dir, 'LICENSE'), os.path.join(working_dir, 'glmtools_docs', 'LICENSE'))
        shutil.move(os.path.join(working_dir, 'MANIFEST.in'), os.path.join(working_dir, 'glmtools_docs', 'MANIFEST.in'))
        shutil.move(os.path.join(working_dir, 'README.md'), os.path.join(working_dir, 'glmtools_docs', 'README.md'))
        shutil.move(os.path.join(working_dir, 'setup.py'), os.path.join(working_dir, 'glmtools_docs', 'setup.py'))

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        working_dir = os.getcwd()
        os.makedirs(os.path.join(working_dir, 'glmtools_docs'), exist_ok = True)
        shutil.move(os.path.join(working_dir, 'docs'), os.path.join(working_dir, 'glmtools_docs', 'docs'))
        shutil.move(os.path.join(working_dir, 'examples'), os.path.join(working_dir, 'glmtools_docs', 'examples'))
        shutil.move(os.path.join(working_dir, 'environment.yml'), os.path.join(working_dir, 'glmtools_docs', 'environment.yml'))
        shutil.move(os.path.join(working_dir, 'LICENSE'), os.path.join(working_dir, 'glmtools_docs', 'LICENSE'))
        shutil.move(os.path.join(working_dir, 'MANIFEST.in'), os.path.join(working_dir, 'glmtools_docs', 'MANIFEST.in'))
        shutil.move(os.path.join(working_dir, 'README.md'), os.path.join(working_dir, 'glmtools_docs', 'README.md'))
        shutil.move(os.path.join(working_dir, 'setup.py'), os.path.join(working_dir, 'glmtools_docs', 'setup.py'))
        
setup(
    name='glmtools',
    version='0.1dev',
    description='Python tools for reading, processing, and visualizing GOES Geostationary Lightning Mapper data',
    packages=find_packages(),# ['glmtools',],
    author='Eric Bruning',
    author_email='eric.bruning@gmail.com',
    url='https://github.com/deeplycloudy/glmtools/',
    license='BSD-3-Clause',
    long_description=open('README.md').read(),
    cmdclass={'install': PostInstallCommand, 
              'develop': PostDevelopCommand,},
    include_package_data=True,
)
