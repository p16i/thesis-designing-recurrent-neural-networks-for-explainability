#!/bin/bash

pip install jupyter

# install extension package
pip install jupyter_contrib_nbextensions

# install js and css stuffs
jupyter contrib nbextension install --user

# enable plugins
jupyter nbextension enable toc2/main
