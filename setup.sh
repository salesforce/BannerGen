#! /bin/bash
# Setup BannerGen headless Chrome browser and reinstall cv2 libraries

pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python
pip install 'opencv-python-headless<4.3'
ln -fs /usr/share/zoneinfo/America/Los_Angelos /etc/localtime
cp executables/chromedriver /usr/bin/chromedriver
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install ./executables/google-chrome-stable_current_amd64.deb