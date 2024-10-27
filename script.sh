#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 01:00:00
#SBATCH -A m4704
#SBATCH --output=logs/tricount_output_batch_1.log
#SBATCH --error=logs/tricount_error_batch_1.log

#OpenMP settings:
export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# Add any other SLURM directives you need

# Directory containing the graph datasets
DATASET_DIR="/pscratch/sd/h/hsarikan/ml/inputs/edited/batch_1"

# Output CSV file
OUTPUT_CSV="/pscratch/sd/h/hsarikan/ml/results.csv"

# Your executable
EXECUTABLE="/pscratch/sd/h/hsarikan/ml/code/tricount.cuda"

# Ensure the output CSV exists and has headers
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Dataset,Vertices Count,Edges Count, Max_d, Avg_d ,Longest Shortest Path,Time_64,Time_128,Time_256,Time_512,Time_1024,Tricount" > "$OUTPUT_CSV"
fi

# Iterate over all files in the dataset directory
for dataset in "$DATASET_DIR"/*; do
    if [ -f "$dataset" ]; then
        echo "Processing dataset: $dataset"
        
        # Run your program
        $EXECUTABLE "$dataset" "$OUTPUT_CSV"
        
        echo "Finished processing $dataset"
        echo "----------------------------"
    fi
done

echo "All datasets processed."