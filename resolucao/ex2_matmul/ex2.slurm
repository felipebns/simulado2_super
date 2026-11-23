#!/bin/bash
#SBATCH --job-name=ex2_matmul
#SBATCH --output=ex2.out
#SBATCH --partition=merry_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=4G

echo "========= GPU simples ========="
./simples
echo "========= GPU tiling ========="
./tiling
