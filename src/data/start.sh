#!/bin/bash
python src/data/make_dataset.py data/raw data/processed
python src/data/make_norm_dataset.py 
