#! /bin/bash
# Setup BannerGen headless Chrome browser and Python libraries

ln -fs /usr/share/zoneinfo/America/Los_Angelos /etc/localtime
cp executables/chromedriver /usr/bin/chromedriver
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install ./executables/google-chrome-stable_current_amd64.deb
#pip install -r requirements.txt
#pip uninstall -y opencv-python opencv-python-headless
#pip install opencv-python
#pip install 'opencv-python-headless<4.3'
#pip install -r requirements.txt
#pip uninstall -y opencv-python opencv-python-headless
#pip install opencv-python
#pip install 'opencv-python-headless<4.3'