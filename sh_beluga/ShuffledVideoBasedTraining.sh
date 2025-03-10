#!/bin/bash
#SBATCH --account=def-###
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4                
#SBATCH --tasks-per-node=4 
#SBATCH --cpus-per-task=4  
#SBATCH --mem=128G  
#SBATCH --time=0-03:58:59


# go to the directory where the job will run
# cd $SLURM_SUBMIT_DIR
# activate python virtual environment
module load StdEnv/2023 python/3.11.5
source /home/###/projects/def-###/###/py3115Torch250Env/bin/activate


export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script with caseFlag: ${1}"

# The $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python test/test_training_ResNet_Scenario_DDP.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)) --num_epochs 100 --deviceArg cuda  > log/Scenario_ddpResNet_Train_${SLURM_JOBID}.log



