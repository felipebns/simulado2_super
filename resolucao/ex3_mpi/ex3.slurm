#!/bin/bash
#SBATCH --job-name=ex3_mpi
#SBATCH --output=ex3.out
#SBATCH --partition=merry_cpu
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

echo "========= Coletivo (base) ========="
srun --mpi=pmix ./base
echo "========= Ponto a ponto não bloqueante ========="
srun --mpi=pmix ./nao_bloqueante
