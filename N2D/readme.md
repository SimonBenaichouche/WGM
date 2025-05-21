
# Navier Stokes 2D
This folder contains python code to run navier stokes experiments of the paper

intructions: 
you need to modify data path in the file run.py

the file allow to parameterize the gradient field as the gradient of a scalar function (mandatory if you want to train the surrogate model but not for WGM).

the file conv_score_matching contains nn architecture and lightning implementation of the model. 
