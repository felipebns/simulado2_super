# Exercício 2 — Multiplicação de matrizes em GPU: simples e tiling (3 pts)

Arquivos: `resolucao/ex2_matmul/` → `base.cpp` (enunciado), `simples.cu`, `tiling.cu`, `ex2.slurm`.

| Rubrica | Onde está |
|---|---|
| Paralelismo simples (kernel + cópias CPU↔GPU) | `simples.cu` |
| Paralelismo com tiling (memória compartilhada) | `tiling.cu` |

Referência do enunciado (CPU, Franky): `Tempo: 71.6775 s`, `C[0][0] = 2170.03`,
`C[N/2][N/2] = 2190.39`, `C[N-1][N-1] = 2218.17`.

---

## Passo 1 — Achar o laço caro

`multiplicacaoCPU` tem 3 laços (`i`, `j`, `k`). Cada `C[i][j]` é independente dos outros, então
**cada thread calcula um `C[i][j]`** e o laço em `k` fica dentro da thread, igual à CPU (mesma
ordem de soma → mesmo resultado).

## Passo 2 — Versão simples (`simples.cu`)

Kernel = o corpo dos laços `i, j` da CPU, com `i, j` vindos dos índices da thread:
```cpp
int i = blockIdx.y * blockDim.y + threadIdx.y;   // linha
int j = blockIdx.x * blockDim.x + threadIdx.x;   // coluna
if (i >= N || j >= N) return;
double soma = 0.0;
for (int k = 0; k < N; k++) {
    double val = A[i * N + k] * B[k * N + j];
    soma += val * sin(val) + cos(val);
}
C[i * N + j] = soma;
```
- `threadIdx.x → j` (coluna): threads vizinhas leem `B[k][j]` e escrevem `C[i][j]` em posições
  vizinhas → acesso coalescido.
- Grid 2D: `dim3 threads(16, 16); dim3 blocos((N+15)/16, (N+15)/16);`

No `main`, o mesmo da base mais o roteiro padrão:
```cpp
cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);   // antes do t0
t0 = now();
cudaMemcpy(d_A, A.data(), bytes, H2D);  cudaMemcpy(d_B, B.data(), bytes, H2D);
multiplicacaoGPU<<<blocos, threads>>>(d_A, d_B, d_C, N);
cudaDeviceSynchronize();
cudaMemcpy(C.data(), d_C, bytes, D2H);
t1 = now();
... mesmos prints ...
cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
```
O tempo medido inclui as cópias, para a comparação com a CPU ser justa.

## Passo 3 — Versão com tiling (`tiling.cu`)

Ideia (aula "Otimização de memória em GPU"): cada bloco calcula um tile 16×16 de `C`. Em vez de
cada thread ler `N` valores de A e `N` de B da memória global, o bloco **copia juntos** um tile de
A e um de B para a **memória compartilhada** e todas as threads reutilizam esses valores.

```cpp
__shared__ double As[TILE][TILE];
__shared__ double Bs[TILE][TILE];
int i = blockIdx.y * TILE + ty, j = blockIdx.x * TILE + tx;
double soma = 0.0;
for (int t = 0; t < (N + TILE - 1) / TILE; t++) {       // percorre os tiles ao longo de k
    As[ty][tx] = A[i][t*TILE + tx];                     // cada thread carrega 1 de A
    Bs[ty][tx] = B[t*TILE + ty][j];                     // e 1 de B (0.0 se fora da matriz)
    __syncthreads();                                    // tile completo antes de usar
    for (int k = 0; k < min(TILE, N - t*TILE); k++) {
        double val = As[ty][k] * Bs[k][tx];
        soma += val * sin(val) + cos(val);
    }
    __syncthreads();                                    // todos usaram antes de sobrescrever
}
C[i][j] = soma;
```
Cuidados:
- **2 barreiras**: uma depois de carregar e outra antes de carregar o próximo tile.
- **`kmax = min(TILE, N - t*TILE)`**: aqui não dá para só preencher com 0 como num produto comum,
  porque `cos(0) = 1` somaria 1 a mais. Com N = 2048 (múltiplo de 16) isso nem acontece, mas
  deixa o kernel certo para qualquer N.
- A ordem da soma em `k` continua 0, 1, 2, …, então o resultado é o mesmo da versão simples.
- Leituras da memória global por thread: 2·N = 4096 na simples → 2·N/16 = 256 no tiling (16× menos).

## Passo 4 — Compilar e rodar

```bash
module load cuda/12.8.1
nvcc -O3 -arch=sm_86 simples.cu -o simples
nvcc -O3 -arch=sm_86 tiling.cu -o tiling
sbatch ex2.slurm        # depois: cat ex2.out
# CPU (referência): g++ -O3 base.cpp -o base && srun --partition=merry_cpu --mem=1G ./base
```

## Passo 5 — Resultados (Franky, RTX A1000)

| Versão | Tempo | Speedup vs CPU |
|---|---|---|
| CPU (enunciado) | 71,68 s | 1× |
| CPU (meu `srun`, `g++ -O3`) | 143,65 s (mesmos valores) | — |
| GPU simples (16×16) | 5,53 s | 13× |
| GPU tiling 16×16 | 5,53 s | 13× |
| tiling 8×8 / 32×32 (teste) | 5,54 s / 5,58 s | — |

Os valores de validação saem **idênticos** aos do enunciado nas duas versões
(2170.03 / 2190.39 / 2218.17). O meu tempo de CPU deu o dobro do enunciado, provavelmente porque
o job caiu num núcleo E (eficiência) do i7-13700T; o speedup acima usa o valor do enunciado.

**Por que o tiling não ficou mais rápido (o que escrever na entrega):**
- O tiling reduz leituras da memória global. Ele só acelera se a memória for o gargalo.
- Aqui o gargalo é **cálculo em `double`**: cada iteração faz `sin` e `cos` em `double`, o que dá
  ~34 instruções FP64 (contei no assembly: `cuobjdump -sass simples` mostra 24 DFMA + 8 DMUL +
  2 DADD + 2 DSETP no kernel). A RTX A1000 faz FP64 a 1/64 da taxa de FP32.
- Conta: 2048³ = 8,6·10⁹ iterações em 5,5 s = 1,56·10⁹ it/s × 34 ≈ 53·10⁹ instr. FP64/s, que é o
  **pico de FP64** da placa (18 SMs × 2 unidades FP64 × 1,462 GHz ≈ 52,6·10⁹/s). A GPU já está no
  limite de cálculo nas duas versões, então trocar o caminho dos dados não muda nada.
- Na versão simples, os acessos já são bem comportados (A é o mesmo endereço para a linha do warp,
  B e C coalescidos), e o cache L1/L2 resolve o reúso que o tiling faria.
- Para provar com o `ncu` (seção SpeedOfLight), no Franky deu `ERR_NVGPUCTRPERM`: o usuário não
  tem permissão para os contadores da GPU.
