#!/bin/bash
#SBATCH --job-name=gnn_training
#SBATCH --partition=gpuq-a30
#SBATCH --nodelist=gpu[001]
#SBATCH --output=gnn_sampled_yenib1_train_output_%j.txt
#SBATCH --error=gnn_sampled_yenib1_train_error_%j.txt
#SBATCH --gres=gpu:1

echo "Starting SLURM script"

#OpenMP settings:
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Activate your conda environment if needed
#source activate gnn_cuda
source ~/miniconda3/bin/activate gnn_cuda
echo $CONDA_DEFAULT_ENV


# Run the python script
EXECUTABLE="/home/arikan/ml_cuda/gnn/gnn_with_sampling.py"

# Print working directory and list files
pwd
# ls -l $EXECUTABLE

#python $EXECUTABLE --mode train
python $EXECUTABLE --mode train --batch 1

# Keep the job running for a moment to ensure output is written
# sleep 5

echo "Script execution completed"