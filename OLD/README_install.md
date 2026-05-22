# Requirements
Install the requirements.

    pip3 install -r requirements.txt

# Packaging

Download the source code

    git clone https://github.com/trucomanx/SectorPooling

The next command generates the `dist/SectorPooling-VERSION.tar.gz` file.

    cd SectorPooling/src
    python setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/SectorPooling-VERSION.tar.gz
