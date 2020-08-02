brain-python-interface (a.k.a. bmi3d)
====================================
**This the unstable python 3 branch. It may not yet do what you want it to do. Use the master branch for the heavily-tested (but out of date) python 2 version**

This package contains Python code to run electrophysiology behavioral tasks,
with emphasis on brain-machine interface (BMIs) tasks. This package differs 
from other similar packages (e.g., BCI2000--http://www.schalklab.org/research/bci2000) 
in that it is primarily intended for intracortical BMI experiments. 

This package has been used with the following recording systems:
- Omniplex neural recording system (Plexon, Inc.). 
- Blackrock NeuroPort

Code documentation can be found at http://carmenalab.github.io/bmi3d_docs/

Getting started
---------------
# Dependencies
## Linux/OS X
```bash
sudo xargs apt-get -y install < requirements.system
```

## Windows
Visual C++ Build tools (for the 'traits' package)


# Installation
```bash
git clone -b develop https://github.com/carmenalab/brain-python-interface.git
cd brain-python-interface
pip3 install -r requirements.txt
pip3 install -e .
```

## Installation in Docker
- Set up docker on Ubuntu following these instructions: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
- In the `install` folder, execute `build_docker.sh` to build the image
- Also in the `install` folder, execute `run_docker.sh` to load the image. Annoyingly at this time, because the source directory is mounted in a way that is volatile, the bash shell will 'reinstall' the bmi3d package every time you load the image

Graphics generation from inside the image has only been tested in Ubuntu. To test, load the image using `run_docker.sh` and
```bash
cd /src/tests/unit_tests/
python3 test_built_in_vfb_task.py
```
If successful, you'll see the pygame window pop up looking like a poorly-made video game. If unsuccessful, you'll see the graphics in the terminal itself in ASCII art. 


# Setting up the database
```bash
cd db
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py makemigrations tracker
python3 manage.py migrate                  # make sure to do this twice!
```

# Start server
```bash
python3 manage.py runserver
```

# Setup paths and configurations
Once the server is running, open up Chrome and navigate to localhost:8000/setup
- Under "Subjects", make sure at least one subject is listed. A subject named "test" is recommended for separating exploration/debugging from real data. 
- Under "tasks", add a task to the system by giving it the python path for your task class. See documentation link above for details on how to write a task. There are a couple of built in tasks to help you get started. 

	- For example, you can add the built-in task `riglib.experiment.mocks.MockSequenceWithGenerator` just to check that the user interface works
	- If you want to try something graphical, you can add the built-in task `built_in_tasks.passivetasks.TargetCaptureVFB2DWindow`. This will be a "visual feedback" task in which a cursor automatically does the center-out task, a standard task in motor ephys. 


# Run a task
Navigate to http://localhost:8000/exp_log/ in chrome. Then press 'Start new experiment' and run your task. 


# Troubleshooting
This package has a lot of dependencies which makes installation somewhat brittle due to versions of different dependencies not getting along. 

- Installation in a virtual environment (see `venv` in python3) or in a Docker container is recommended to try to isolate the package from version conflict issues. 
- Run scripts in `tests/unit_tests/` to try to isolate which components may not be working correctly. Issues in `riglib` will be easier to fix than issues in the database. 

