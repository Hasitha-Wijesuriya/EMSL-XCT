#!/bin/csh

#SBATCH --account <account>                    # charged account
#SBATCH --time 0:21                           # 31 hour 0 minute time limit
#SBATCH --nodes 2                              # 2 nodes
#SBATCH --ntasks-per-node 36                   # 36 processes on each node
#SBATCH --job-name my_job_name                 # job name in queue (``squeue``)
#SBATCH --error my_job_name-%j.err             # stderr file with job_name-job_id.err
#SBATCH --output my_job_name-%j.out            # stdout file
#SBATCH --mail-user=hasitha.wijesuriya@pnnl.gov  # email user
#SBATCH --mail-type END                        # when job ends

module purge                                   # removes the default module set
module load intel
module load impi

mpirun -n 72 ./my_exe