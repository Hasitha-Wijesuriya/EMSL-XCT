#!/bin/bash

#SBATCH --account emsl61599                    # charged account
#SBATCH --gres=<gpu:2>
#SBATCH --time 120                           # 2 hour 0 minute time limit
#SBATCH --nodes 1                              # 1 nodes
#SBATCH --job-name mbirjax                 # job name in queue (``squeue``)
#SBATCH --error my_job_name-%j.err             # stderr file with job_name-job_id.err
#SBATCH --output my_job_name-%j.out            # stdout file
#SBATCH --mail-user=hasitha.wijesuriya@pnnl.gov  # email user
#SBATCH --mail-type END                        # when job ends

source ~/venv/anaconda3/etc/profile.d/conda.sh                                 # removes the default module set
conda deactivate
conda deactivate
conda activate emsl-xct

srun python /home/wije370/EMSL-XCT/backend/mbir_script.py
