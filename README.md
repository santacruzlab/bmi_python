brain-python-interface (a.k.a. bmi3d)
====================================
**This is the unstable python 3 branch under development.** 

This package contains Python code to run brain-machine interface (BMIs) tasks as well as other neurophysiology experiments. This package differs from other similar packages (e.g., BCI2000--http://www.schalklab.org/research/bci2000) in that it is primarily intended for intracortical BMI experiments. 

This package was originally written for Python 2. The code has been updated to run with Python 3, but is currently unstable.

HTML documentation for the code can be found at http://carmenalab.github.io/bmi3d_docs/

Getting started 
---------------
# Dependencies
## Linux/OS X
(none at this time)

## Windows
Visual C++ Build tools (for the 'traits' package)

# Installation
```bash
git clone https://github.com/DerekYJC/bmi_python.git
cd brain-python-interface
pip3 install -r requirements.txt
pip3 install -e .
```

# set up the database
```bash
cd db
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py makemigrations tracker
python3 manage.py migrate                  # make sure to do this twice!
```

# start server
```bash
python3 manage.py runserver
```

# Setup
Once the server is running, open up Chrome and navigate to localhost:8000/setup
- Under 'subjects', make sure at least one subject is listed. A subject named 'test' is recommended for separating exploration/debugging from real data. 
- Under 'tasks', add a task to the system by giving it the python path for your task class. See documentation link above for details on how to write a task. There are a couple of built in tasks to help you get started. 

	- For example, you can add the built-in task 'riglib.experiment.mocks.MockSequenceWithGenerator' just to check that the user interface works
	- If you want to try something graphical, you can add the built-in task 'built_in_tasks.passivetasks.TargetCaptureVFB2DWindow'. This will be a 'visual feedback' task in which a cursor automatically does the center-out task, a standard task in motor ephys. 

