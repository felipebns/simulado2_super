#!/bin/bash
#SBATCH --job-name=ex1_prefix
#SBATCH --output=ex1.out
#SBATCH --partition=merry_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=2G

echo "========= CPU (base) ========="
./base
echo "========= GPU síncrono ========="
./sincrono
echo "========= GPU assíncrono ========="
./assincrono

# A 1ª execução de cada binário é mais lenta (aquecimento): repete para medir
echo "========= Tempos (5 execuções de cada) ========="
for prog in base sincrono assincrono; do
    for i in 1 2 3 4 5; do ./$prog | grep Tempo; done
done
