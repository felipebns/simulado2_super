# Simulado 2 — passo a passo

Enunciado: `../Simulado2.html`. Só usei o que está nas aulas: kernels CUDA, `cudaMalloc`/`cudaMemcpy`,
memória compartilhada, tiling, scan, streams, memória pinned, MPI `Isend`/`Irecv`/`Wait`/`Waitall`
e Slurm.

| Exercício | Arquivos (`../resolucao/`) | Explicação |
|---|---|---|
| 1. Prefix sum + normalização em GPU (4,0) | `ex1_prefix_sum/` | [1_prefix_sum_cuda.md](1_prefix_sum_cuda.md) |
| 2. Matmul em GPU: simples e tiling (3,0) | `ex2_matmul/` | [2_matmul_tiling.md](2_matmul_tiling.md) |
| 3. MPI ponto a ponto não bloqueante (3,0) | `ex3_mpi/` | [3_mpi_nao_bloqueante.md](3_mpi_nao_bloqueante.md) |

Cada pasta tem o `base.cpp` do enunciado (referência de saída e tempo), a solução e um `.slurm`.

---

## 0. Como rodar tudo no Franky

```bash
# no notebook (rede do Insper ou VPN)
scp -r resolucao <usuario>@10.103.1.38:~/simulado2
ssh <usuario>@10.103.1.38

# no head-node: compila
module load cuda/12.8.1
cd ~/simulado2/ex1_prefix_sum
g++ -O3 base.cpp -o base
nvcc -O3 -arch=sm_86 sincrono.cu -o sincrono
nvcc -O3 -arch=sm_86 assincrono.cu -o assincrono
sbatch ex1.slurm                       # saída em ex1.out

cd ../ex2_matmul
nvcc -O3 -arch=sm_86 simples.cu -o simples
nvcc -O3 -arch=sm_86 tiling.cu -o tiling
sbatch ex2.slurm                       # saída em ex2.out

cd ../ex3_mpi
mpic++ -O2 base.cpp -o base
mpic++ -O2 nao_bloqueante.cpp -o nao_bloqueante
sbatch ex3.slurm                       # saída em ex3.out
```
Acompanhar a fila: `squeue -u <usuario>`. Testar rápido sem slurm:
`srun --partition=merry_gpu --gres=gpu:1 ./sincrono` ou
`srun --partition=merry_cpu --mpi=pmix --mem=1G -n 4 ./nao_bloqueante`.

## Resumo dos resultados (medidos no Franky)

| Exercício | Versão | Tempo | Saída |
|---|---|---|---|
| 1 | CPU base | 0,056 ms | `7.68322e+09` (erro de float) |
| 1 | GPU síncrona | 0,185 ms | `7.68331e+09` = N·(N+1) exato |
| 1 | GPU assíncrona | 0,177 ms | igual à síncrona |
| 2 | CPU (enunciado) | 71,68 s | 2170.03 / 2190.39 / 2218.17 |
| 2 | GPU simples | 5,53 s (13×) | igual ao enunciado |
| 2 | GPU tiling | 5,53 s (13×) | igual ao enunciado |
| 3 | Scatter/Gather e Isend/Irecv | — | `Soma total = 136` (n = 1, 2, 4, 8, 16) |

Os dois resultados "estranhos" (GPU mais lenta que a CPU no ex. 1; tiling igual ao simples no
ex. 2) estão explicados nos arquivos de cada exercício. Na entrega, **explique com os números**.

---

## Receita para um exercício parecido

### A. Portar um laço da CPU para a GPU (versão "simples")
1. Rode a base e guarde a saída e o tempo.
2. Ache o laço caro. **Cada iteração independente vira uma thread** (1 thread por elemento de saída).
   Laço duplo `i, j` → grid 2D com `j = blockIdx.x*blockDim.x + threadIdx.x` (coluna, coalescido)
   e `i = ...y` (linha). Laço interno com dependência (soma em `k`) fica dentro da thread.
3. Roteiro do `main`:
   `cudaMalloc` → `cudaMemcpy H2D` → `kernel<<<(N+T-1)/T, T>>>` → `cudaDeviceSynchronize` →
   `cudaMemcpy D2H` → `cudaFree`. Sempre `if (idx < N)` no kernel.
4. Mantenha os mesmos prints da base para comparar a saída.

### B. Tiling (qualquer `C[i][j] = Σk f(A[i][k], B[k][j])`)
- `__shared__ T As[TILE][TILE], Bs[TILE][TILE]`; laço `t` sobre os tiles de `k`; cada thread carrega
  `A[i][t*TILE+tx]` e `B[t*TILE+ty][j]`; `__syncthreads()` **depois de carregar** e **depois de usar**.
- Fora da matriz: carregue 0, mas cuidado se `f(0,0) ≠ 0` (aqui `cos(0) = 1`): limite o laço
  interno em `min(TILE, N - t*TILE)`.
- Meça TILE = 8, 16, 32. Se não melhorar, verifique se o gargalo é cálculo (conta de instruções ×
  pico da GPU), como no ex. 2.

### C. Redução/scan com mais de um bloco
1. Kernel por bloco em memória compartilhada (código da aula); a última thread (scan) ou a thread 0
   (redução) grava o resultado do bloco em `somas[blockIdx.x]`.
2. Combina os resultados dos blocos: o mesmo kernel com 1 bloco sobre `somas` (funciona se
   nº de blocos ≤ 1024).
3. Scan: soma `somas[b-1]` em cada elemento do bloco `b`.
4. Se um kernel precisa de um valor que outro bloco ainda pode estar escrevendo (ex.: `prefix[N-1]`),
   guarde esse valor num buffer separado (`d_total`).

### D. Versão assíncrona (streams)
1. Host com `cudaMallocHost` (pinned), libera com `cudaFreeHost`.
2. `cudaStreamCreate` para 2 streams; divide os dados em 1 pedaço por stream.
3. Por pedaço: `cudaMemcpyAsync H2D` → `kernel<<<b, t, 0, stream>>>` → `cudaMemcpyAsync D2H`,
   tudo na **mesma stream**: a ordem na stream garante as dependências do pedaço.
4. Trabalho de CPU (gerar/ler dados) **entre** os lançamentos, para sobrepor com a GPU.
5. Dependência **entre** streams (ex.: precisa de todos os pedaços): `cudaStreamSynchronize` em
   todas antes do próximo passo.
6. `cudaStreamSynchronize` em todas antes de a CPU ler o resultado; `cudaStreamDestroy` no fim.
7. Meça com mais e menos pedaços: pedaços demais = lançamentos demais = mais lento.

### E. Coletiva MPI → ponto a ponto não bloqueante

| Coletiva | Rank 0 | Outros ranks |
|---|---|---|
| `Scatter` | `Isend(dados + p*n)` para p ≥ 1; copia o bloco 0 | `Irecv` + `Wait` |
| `Gather` | `Irecv(&v[p])` de p ≥ 1; `v[0]` = o próprio; `Waitall` | `Isend` + `Wait` |
| `Bcast` | `Isend(mesmo buffer)` para p ≥ 1; `Waitall` | `Irecv` + `Wait` |
| `Reduce` | como `Gather` e depois soma | `Isend` + `Wait` |

Um `MPI_Request` por mensagem; `Wait`/`Waitall` **antes** de ler o que foi recebido e antes de
alterar o que foi enviado; tags diferentes para cada tipo de mensagem.

### F. Slurm
GPU:
```bash
#SBATCH --partition=merry_gpu
#SBATCH --gres=gpu:1        # pede 1 GPU
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
```
MPI: `--partition=merry_cpu`, `--nodes=2`, `--ntasks=4`, `--mem=1G` e `srun --mpi=pmix ./prog`
dentro do script. Limites: `merry_*` 5 min, `sunny_*` 20 min, `pluton_*` 1 h.

### Armadilhas que apareceram
- `nvcc` sem `-arch=sm_86` gera código genérico → compilação JIT de ~120 ms no 1º kernel, dentro
  do tempo medido.
- 1ª execução de um binário novo é mais lenta → repita e use a mediana.
- `srun` sem `--mem` → job preso com `(MaxMemPerLimit)`.
- `nsys` só existe no nó depois de `module load cuda/12.8.1`; `ncu` dá `ERR_NVGPUCTRPERM` (sem
  permissão para os contadores).
- `float` somado em ordem diferente dá resultado diferente: compare com o valor exato, se houver.
