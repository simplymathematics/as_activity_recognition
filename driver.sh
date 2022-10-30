#!/bin/bash
dvc pull || dvc repro --force
dvc repro
cd results && python -m http.server 8000
