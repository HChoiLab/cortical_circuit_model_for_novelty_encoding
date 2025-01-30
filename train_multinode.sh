#!/bin/bash

# Default values for the arguments
node_rank=0
num_runs=1
action=false

# Parse optional arguments
while getopts "n:r:a" opt; do
  case $opt in
    n)
      node_rank=$OPTARG
      ;;
    r)
      num_runs=$OPTARG
      ;;
    a)
      action=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Shift off the options and optional --
shift $((OPTIND-1))

# Get the node names for this slurm job
raw_nodelist=$(scontrol show job $SLURM_JOBID | awk -F'=' '/NodeList/ && !/ReqNodeList|ExcNodeList/ {print $2}')
IFS=',' read -r -a nodes <<< "$raw_nodelist"

current_node=$(hostname)
target_node="${nodes[$node_rank]}"


# Define the training command
train_cmd_no_action="
module load anaconda3/2022.05
conda activate projects
cd $HOME/code/cortical_circuit_model_for_novelty_encoding
rm -rf ~/.local/share/Trash/*
python distributed_training.py --num_runs $num_runs
"

train_cmd_action="
module load anaconda3/2022.05
conda activate projects
cd $HOME/code/cortical_circuit_model_for_novelty_encoding
rm -rf ~/.local/share/Trash/*
python distributed_training.py --num_runs $num_runs --action
"

if [[ "$action" = true ]]; then
    for node in "${nodes[@]}"; do
        echo "Training on node: $node"
        if [[ "$node" == "$current_node" ]]; then
            eval "$train_cmd_action" &
        else
            ssh "$node" "$train_cmd_action" &
        fi
    done
    wait
else
    for node in "${nodes[@]}"; do
        echo "Training on node: $node"
        if [[ "$node" == "$current_node" ]]; then
            eval "$train_cmd_no_action" &
        else
            ssh "$node" "$train_cmd_no_action" &
        fi
    done
    wait
fi
