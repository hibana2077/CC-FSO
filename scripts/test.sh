#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=50GB           
#PBS -l walltime=00:00:20  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load python3/3.10.0
module load cuda/12.6.0

nvidia-smi >> gpu-info.txt

cd ..

source /scratch/rp06/sl5952/CC-FSO/.venv/bin/activate
python3 exp/test.py >> out1.txt

