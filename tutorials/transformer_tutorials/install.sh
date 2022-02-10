#!/usr/bin/env bash

#Creates virtual environment and installs dependencies

python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
