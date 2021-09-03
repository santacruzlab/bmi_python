#!/bin/bash
####### Declare environment variables
CODE=$HOME
BMI3D=$CODE/bmi_python ### Directory in which to install the bmi3d software


####### Set up directories
mkdir -p $CODE
sudo mkdir /backup
sudo chown $USER /backup

sudo mkdir /storage
sudo chown -R $USER /storage
mkdir /storage/plots
mkdir $CODE/src/

sudo apt-get -y install git gitk
if [ ! -d "$HOME/code/bmi3d" ]; then
    git clone https://github.com/carmenalab/bmi3d.git $HOME/code/bmi3d

    #Add tasks & analysis, if desired
    git clone https://github.com/carmenalab/bmi3d_tasks_analysis.git $HOME/code/bmi3d_tasks_analysis

    #Make symlinks to tasks/analysis in main bmi3d repository
    ln -s $HOME/code/bmi3d_tasks_analysis/analysis $HOME/code/bmi3d/analysis
	ln -s $HOME/code/bmi3d_tasks_analysis/tasks $HOME/code/bmi3d/tasks

fi

# make log directory
mkdir $BMI3D/log

# SRS did not reinstall this, seemed to be working
####### Reconfigure Ubuntu package manager
sudo apt-add-repository "deb http://www.rabbitmq.com/debian/ testing main"
cd $HOME
wget http://www.rabbitmq.com/rabbitmq-signing-key-public.asc
sudo apt-key add rabbitmq-signing-key-public.asc
sudo add-apt-repository ppa:webupd8team/sublime-text-2

## Refresh the package manager's list of available packages
sudo apt-get update


####### Install Ubuntu dependencies
sudo apt-get install python3.8-dev
sudo apt-get -y install python-pip libhdf5-serial-dev
sudo apt-get -y install python-numpy
sudo apt-get -y install python-scipy
# setup the CIFS, SRS error with smbfs
sudo apt-get -y install smbclient cifs-utils smbfs
# matplotlib
sudo apt-get -y install python-matplotlib
# pygame: SRS >>python3 -m pip install -U pygame --user
sudo apt-get -y install mercurial python-dev python-numpy ffmpeg libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
# install tools
sudo apt-get -y install libtool automake bison flex
# ssh
sudo apt-get -y install openssh-server
# text editors
sudo apt-get -y install sublime-text vim-gnome
sudo apt-get -y install rabbitmq-server
sudo apt-get -y install libusb-dev
sudo apt-get -y install ipython
# NIDAQ
sudo apt-get -y install libcomedi-dev
sudo apt-get -y install python-comedilib
sudo apt-get -y install swig
# DHCP server
sudo apt-get -y install isc-dhcp-server
# cURL: command line utility for url transfer
sudo apt-get -y install curl
sudo apt-get -y install sqlite3
# Arduino IDE
sudo apt-get install arduino arduino-core  
# Serial lib
sudo apt-get install setserial

####### Install Python dependencies
sudo pip3 install numexpr 
sudo pip3 install cython 
sudo pip3 install django-celery 
sudo pip3 install traits 
sudo pip3 install pandas 
sudo pip3 install patsy 
sudo pip3 install statsmodels 
sudo pip3 install PyOpenGL PyOpenGL_accelerate 
sudo pip3 install Django==1.6 # SRS previously had django 3.2.6 installed
sudo pip3 install pylibftdi 
sudo pip3 install nitime 
sudo pip3 install sphinx
sudo pip3 install numpydoc
sudo pip3 install tornado
sudo pip3 install tables  # SRS install 3.6.1 instead of suggested 2.4.0 (incompatible with Python 3)
sudo pip3 install sklearn
sudo pip3 install anyjson
sudo pip3 install billiard


####### Download any src code
git clone https://github.com/sgowda/plot $HOME/code/plotutil
git clone https://github.com/sgowda/robotics_toolbox $HOME/code/robotics
# pygame: SRS did not run this, error
hg clone https://bitbucket.org/pygame/pygame $HOME/code/pygame
# Phidgets code: SRS not using phidgets, didn't do this
wget http://www.phidgets.com/downloads/libraries/libphidget.tar.gz
wget http://www.phidgets.com/downloads/libraries/PhidgetsPython.zip




####### Install source code, configure software
# plexread module: SRS ValueError: 'plexon/psth.pyx' doesn't match any files
cd $BMI3D/riglib
sudo python3 setup.py install

# pygame: SRS didn't do this since bitbucket didn't clone
cd $HOME/code/pygame
sudo python setup.py install

# symlink for iPython
sudo ln -s /usr/bin/ipython /usr/bin/ipy

# NIDAQ software -- deprecated!
# $HOME/code/bmi3d/riglib/nidaq/build.sh

# Phidgets libraries: SRS didn't do this since we aren't using phidgets
cd $CODE/src/
tar xzf libphidget.tar.gz 
cd libphidget*
./configure
make
sudo make install

cd $CODE/src/
unzip PhidgetsPython.zip  
cd PhidgetsPython
sudo python setup.py install



####### Configure udev rules, permissions
# Phidgets: SRS didn't do this since we aren't using phidgets
sudo cp $CODE/src/libphidget*/udev/99-phidgets.rules /etc/udev/rules.d
sudo chmod a+r /etc/udev/rules.d/99-phidgets.rules
# NIDAQ: SRS didn't do this since we aren't using NIDAQ
sudo cp $HOME/code/bmi3d/install/udev/comedi.rules /etc/udev/rules.d/
sudo chmod a+r /etc/udev/rules.d/comedi.rules 
sudo udevadm control --reload-rules
# Group permissions
sudo usermod -a -G iocard $USER # NIDAQ card belongs to iocard group - SRS didn't do this since we aren't using NIDAQ
sudo usermod -a -G dialout $USER # Serial ports belong to 'dialout' group


####### Reconfigure .bashrc
#sed -i '$a export PYTHONPATH=$PYTHONPATH:$HOME/code/robotics' $HOME/.bashrc
sed -i '$a export BMI3D=/home/samantha/bmi_python' $HOME/.bashrc
sed -i '$a source $HOME/bmi_python/pathconfig.sh' $HOME/.bashrc
source $HOME/.bashrc

sudo chown -R $USER ~/.matplotlib # SRS directory doesn't exist

cd $BMI3D/db
python3 manage.py syncdb
# Add superuser 'lab' with password 'lab'


