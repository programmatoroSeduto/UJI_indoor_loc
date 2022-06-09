#! /bin/bash

set -e

echo " --- sudo UPDATE --- "
sudo apt-get update -y
echo " ---===--- "

echo " --- sudo UPGRADE --- "
sudo apt-get upgrade -y
echo " ---===--- "

echo " --- PY VERISON --- "
python3 --version
echo " ---===--- "

echo " --- pip VERISON --- "
pip3 --version
# sudo apt install -y python3-pip
echo " ---===--- "

echo " --- setup: numpy ---"
pip3 install numpy
echo " ---===--- "

echo " --- setup: matplotlib ---"
pip3 install matplotlib
echo " ---===--- "

echo " --- setup: pandas ---"
pip3 install pandas
echo " ---===--- "

echo " --- setup: scikitlearn ---"
pip3 install sklearn
echo " ---===--- "
