#!/bin/bash
source env/bin/activate || { python3 -m venv env; source env/bin/activate; }
python -m pip install -r requirements.txt
python -m pip install dvc-gdrive
python -m pip install protobuf==3.20.0
dvc pull || { dvc repro; dvc push; }
cd results && python -m http.server 8000
