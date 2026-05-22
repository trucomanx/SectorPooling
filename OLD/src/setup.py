#!/usr/bin/python

from setuptools import setup, find_packages
import os 

def func_read(fname):
    return open(os.path.join(os.path.dirname(__file__),fname)).read();

setup(
    name   ='SectorPooling',
    version='0.1.0',
    author ='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    maintainer='Fernando Pujaico Rivera',
    maintainer_email='fernando.pujaico.rivera@gmail.com',
    url='https://github.com/trucomanx/SectorPooling',
    license='GPLv3',
    description='Sector 4 pooling',
    long_description=func_read('README.txt'),
    packages=find_packages(where='.'),
    #package_dir={'':'./'},
    package_data = {
        'SectorPooling': ['*.py']
    },
    keywords=['pooling','cnn'],
    install_requires=[
       "tensorflow",
       "numpy" #"Django >= 1.1.1",
    ],
)

print("")
print("find_packages(where=\'.\');")
print(find_packages(where='.'))

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/SectorPooling-0.1.0.tar.gz 
