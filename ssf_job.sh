#!/bin/bash
#PBS -N torch-sim-edith
#PBS -l nodes=1
#PGS -l gpus=2
#PBS -l walltime=00:05:00
#PBS -l filesystems=home

source "$HOME/.bashrc"
conda activate torch-sim-edith
python "$HOME/torch-sim-edith/soft_sphere_fire.py"
