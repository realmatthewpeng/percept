#!/usr/bin/env bash

source /Users/MP/Desktop/WI25/CS291A/percept_project/venv/bin/activate

python3 combined_pipeline.py -c _configs/argusIIscoreboard.yaml > argusIIscoreboard.log
python3 combined_pipeline.py -c _configs/argusIIscoreboard2.yaml > argusIIscoreboard2.log
python3 combined_pipeline.py -c _configs/argusIIscoreboard3.yaml > argusIIscoreboard3.log
python3 combined_pipeline.py -c _configs/argusIIaxon2.yaml > argusIIaxon2.log
python3 combined_pipeline.py -c _configs/argusIIaxon3.yaml > argusIIaxon3.log