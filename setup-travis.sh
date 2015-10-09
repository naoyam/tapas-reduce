#/bin/sh -f

# things to do for travis-ci in the before_install section

#install a newer cmake since at this time Travis only has version 2.8.7
sudo add-apt-repository --yes ppa:kalakris/cmake
sudo apt-get update -qq
sudo apt-get install cmake
