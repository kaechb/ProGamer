#!/bin/bash
#SBATCH --partition=allgpu
#SBATCH --constraint='P100'|'V100'|'A100'
#SBATCH --time=48:00:00                           # Maximum time requested
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --chdir=/home/kaechben/slurm        # directory must already exist!
#SBATCH --job-name=hostname
#SBATCH --output=%j.out               # File to which STDOUT will be written
#SBATCH --error=%j.err                # File to which STDERR will be written
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=max.muster@desy.de            # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.
export WANDB_API_KEY=f39ea2cc30c7a621000b7fa3355a8c0e848a91d3
export WANDB_PROJECT="linear"
export WANDB_ENTITY="kaechben"

unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate jetnet2
path=ProGamer
cd /home/$USER/$path/
wandb login f39ea2cc30c7a621000b7fa3355a8c0e848a91d3
# wandb sweep sweep.yaml
# wandb agent $SWEEP_ID

# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
path=ProGamer
POSTFIX=$(date -d "today" +"%d_%H_%M")
echo $POSTFIX
mkdir /beegfs/desy/user/kaechben/code/${POSTFIX}
cp /home/$USER/$path/*.py /beegfs/desy/user/kaechben/code/${POSTFIX}
cp /home/$USER/$path/*.sh /beegfs/desy/user/kaechben/code/${POSTFIX}
cp /home/$USER/$path/*.yaml /beegfs/desy/user/kaechben/code/${POSTFIX}
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
python -u /home/$USER/$path/main.py