print("Starting import test")
print("Importing wandb...")
import wandb
print("Imported wandb.")

# Skipping mpi4py for now to simplify, will add if necessary
# print("Importing mpi4py.MPI...")
# import mpi4py.MPI as MPI
# print("Imported mpi4py.MPI.")

print("Importing yaml...")
import yaml
print("Imported yaml.")

print("Importing logging...")
import logging
print("Imported logging.")

print("Importing os...")
import os
print("Imported os.")

print("Importing config...")
import config # This is vsml-neurips2021-main/config.py
print("Imported config.")

print("Importing numpy...")
import numpy as np
print("Imported numpy.")

print("Importing tensorflow...")
import tensorflow as tf
print("Imported tensorflow.")

print("Importing Experiment...")
from experiment import Experiment # This is vsml-neurips2021-main/experiment.py
print("Imported Experiment.")

print("All imports successful.") 