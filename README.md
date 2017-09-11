# **Behavioral Cloning**
# Udacity's Self-Driving Car NanoDegree - Project 3
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Introduction
---

This repository hosts files for the ***"Behavioral Cloning"*** project for **Udacity's** ***"Self Driving Car Nanodegree"***.

In this project I train a neural network model to drive a car in a simulator. The model is trained on the data collected from the simulator with me in control of the car.

Dependencies
---

1. This project requires python3 and Mac OS X to work.
2. It also requires anaconda - a package dependency and environment manager. Click [here](https://conda.io/docs/download.html) to view instructions on how to install anaconda.
3. In order to run the code you will need to download the ***"Udacity simulator"*** for Mac OS X from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)

Installing required packages
---

1. **Create project environment:** This project comes with an ***environment.yml*** file that lists all the packages required for the project. Running the following command will create a new `conda` environment that is provisioned with all libraries you need to be run this project.
```sh
conda env create -f environment.yml
```
2. **Verify:** Run the following command to verify that the carnd-term1 environment was created in your environments:
```sh
conda info --envs
```
3. **Cleanup:** Run the following command to cleanup downloaded libraries (tarballs, zip files, etc):
```sh
conda clean -tp
```
4. **Activate:** Run the following command to activate the `carnd-term1` environment:
```sh
$ source activate carnd-term1
```
5. **Deactivate:** Run the following command to deactivate the `carnd-term1` environment:
```sh
$ source deactivate
```
6. **Uninstalling:** Run the following command to uninstall the environment:
```sh
conda env remove -n carnd-term1
```

Running the project
---

1. Navigate to the project directory and run the following command:
```sh
python drive.py model.h5
```
2. After the python process is waiting for the simulator to start, start the simulator by navigating to the directory where it was downloaded.

3. In the simulator select the track on the **left** and click on **Autonomous Mode**. Watch the car drive itself.
