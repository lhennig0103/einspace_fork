#!/usr/bin/bash
# Name the job
#SBATCH --job-name=test_einspace
#SBATCH --time=02:00:00
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5GB
#SBATCH --account=p0025023
#SBATCH --output=//hpcwork/p0025023/einspace_fork/slurm_logs/test_run.out%A
# Ask for the maximum memory per CPU
##SBATCH --exclusive
module load GCCcore/.12.2.0
module load Python/3.10.8
source /hpcwork/p0025023/einspace_fork/.env/bin/activate
# export CUDA_VISIBLE_DEVICES=1
# python tools/azure_sweep.py --mode local --config_path configs/sweeps/tinyimagenet/mb_v0.1.yaml
# python3 /hpcwork/p0025023/einspace_fork/einspace/main.py --device cpu --config /hpcwork/p0025023/einspace_fork/configs/addnist/rs_addnist.yaml
# python3 /hpcwork/p0025023/einspace_fork/einspace/main.py --device cuda --config /hpcwork/p0025023/einspace_fork/configs/addnist/rs_addnist.yaml
python3 /hpcwork/p0025023/einspace_fork/einspace/main.py --device cuda --config /hpcwork/p0025023/einspace_fork/configs/cifar100/re_cifar100.yaml
# python3 /hpcwork/p0025023/einspace_fork/einspace/data_utils/generate_indices.py