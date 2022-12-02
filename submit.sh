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
unset LD_PRELOAD
source /etc/profile.d/modules.sh
module purge
module load maxwell gcc/9.3
module load anaconda3/5.2
. conda-init
conda activate jetnet2
wandb login f39ea2cc30c7a621000b7fa3355a8c0e848a91d3

# nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")

# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# echo $nodes
# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
# if [[ "$head_node_ip" == *" "* ]]; then
# IFS=' ' read -ra ADDR <<<"$head_node_ip"
# echo "IFS $IFS"
# echo "ADDR $ADDR"
# if [[ ${#ADDR[0]} -gt 16 ]]; then
#   head_node_ip=${ADDR[1]}
# else
#   head_node_ip=${ADDR[0]}
# fi
# echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
# fi


# echo "head node ip $head_node_ip"
# port=6379
# ip_head=$head_node_ip:$port
# export ip_head
# echo "IP Head: $ip_head"

# echo "Starting HEAD at $head_node"
# srun --nodes=1 --ntasks=1 -w "$head_node" \
#     ray start --temp-dir '/tmp/kaechben/ray' --head --node-ip-address="$head_node_ip" --port=$port \
#    --block & 

# # optional, though may be useful in certain versions of Ray < 1.0.
# sleep 1

# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1 -w "$node_i" \
#         ray start --temp-dir '/tmp/kaechben/ray' --address "$ip_head" \
#          --block & 
#     sleep 5
# done

path=JetNet_NF
POSTFIX=$(date -d "today" +"%d_%H_%M")
echo $POSTFIX
cp /home/$USER/$path/LitJetNet/LitNF/lit_nf.py /beegfs/desy/user/kaechben/code/lit_nf_${POSTFIX}.py
cp /home/$USER/$path/LitJetNet/LitNF/main.py /beegfs/desy/user/kaechben/code/main_${POSTFIX}.py
python -u /home/$USER/$path/LitJetNet/LitNF/main.py