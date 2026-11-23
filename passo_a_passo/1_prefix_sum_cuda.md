# Exercício 1 — Prefix sum + normalização em GPU (4,0 pts)

Arquivos: `resolucao/ex1_prefix_sum/` → `base.cpp` (código do enunciado), `sincrono.cu`,
`assincrono.cu`, `ex1.slurm`.

| Rubrica | Onde está |
|---|---|
| +1,0 GPU síncrona | `sincrono.cu` |
| +2,5 GPU assíncrona (streams, dependências, CPU e GPU ao mesmo tempo) | `assincrono.cu` |
| +0,5 Slurm com GPU | `ex1.slurm` (`--partition=merry_gpu --gres=gpu:1`) |

---

## Passo 1 — Rodar a base e guardar a saída

```bash
g++ -O3 base.cpp -o base && ./base
```
```
[ ... 7.68304e+09 7.68322e+09 ]      <- prefix sum
[ ... 0.999977 1 ]                   <- normalizado
Tempo CPU: ~0,056 ms
```

## Passo 2 — Entender o problema

Tem duas etapas, e uma depende da outra:
1. **Prefix sum** (scan): `prefix[i] = v[0] + ... + v[i]`.
2. **Normalização**: `prefix[i] / total`, com `total = prefix[N-1]`. Só dá para fazer **depois**
   que o scan inteiro terminou.

O scan da aula (`scan_simples`, aula "Redução e Scan com CUDA") só funciona **dentro de um bloco**
(usa memória compartilhada). Como N = 87.654 não cabe em um bloco (máx. 1024 threads), usamos o
esquema de 3 passos: scan por bloco → scan das somas dos blocos → somar o deslocamento (offset).

Exemplo com blocos de 4 elementos:
```
v              = [2 1 3 4 | 5 6 7 8]
1a scan_blocos = [2 3 6 10 | 5 11 18 26]   somas dos blocos = [10, 26]
1b scan somas  = [10, 36]                  -> total = 36
1c offset      = [2 3 6 10 | 15 21 28 36]  (bloco 1 soma 10 = soma do bloco 0)
2  normalizar  = cada valor / 36
```

## Passo 3 — Os kernels (iguais nas duas versões)

Todos usam `THREADS = 1024` e recebem `inicio, fim` (índices globais). Na versão síncrona
`inicio = 0`; a assíncrona usa isso para processar um pedaço por stream.

1. **`scan_blocos`**: é o `scan_simples` da aula com duas mudanças:
   - lê `entrada[gid]` (e não `gid-1`) → scan **inclusivo**, que é o que o enunciado pede;
   - a última thread do bloco grava a soma do bloco: `somas_blocos[gid / THREADS] = XY[tid]`.
2. **Scan das somas**: o **mesmo** kernel, com 1 bloco, sobre `d_somas` (86 valores ≤ 1024).
   A "soma do bloco" desse único bloco é o total → vai para `d_total`.
   ```cpp
   scan_blocos<<<1, THREADS>>>(d_somas, d_somas, d_total, 0, nblocos);
   ```
   Pode ser in-place: cada thread lê a sua posição antes da 1ª barreira e só escreve no fim.
3. **`somar_offset`**: elemento do bloco `b > 0` soma `somas_blocos[b-1]`.
4. **`normalizar`**: `prefix[gid] /= *total`. O total vem de `d_total` (e não de `prefix[N-1]`),
   porque ler `prefix[N-1]` no mesmo kernel que o modifica seria condição de corrida entre blocos.

> Limite: `nblocos ≤ 1024`, ou seja N ≤ 1.048.576. Para N maior, seria preciso mais um nível.

## Passo 4 — Versão síncrona (`sincrono.cu`)

O `main` é o da base, trocando o scan/normalização da CPU por:
```cpp
cudaMalloc(...)                                 // antes do cronômetro
inicio = now();
gera v na CPU
cudaMemcpy(d_v, v, H2D);
scan_blocos<<<nblocos, THREADS>>>(...);         // 1a
scan_blocos<<<1, THREADS>>>(...);               // 1b
somar_offset<<<nblocos, THREADS>>>(...);        // 1c
normalizar<<<nblocos, THREADS>>>(...);          // 2
cudaDeviceSynchronize();
cudaMemcpy(prefix, d_prefix, D2H);  cudaMemcpy(&total, d_total, D2H);
fim = now();
```
Os kernels estão todos na stream padrão, então rodam em ordem, um depois do outro. Os prints
são os mesmos da base.

## Passo 5 — Versão assíncrona (`assincrono.cu`)

Mesmos kernels, só muda a orquestração (como no `otimizado.cu` da aula "Programação Assíncrona"):

1. **Memória pinned** para `v` e `prefix`: `cudaMallocHost` / `cudaFreeHost`. Sem isso o
   `cudaMemcpyAsync` não é assíncrono de verdade.
2. **2 streams**, e o vetor dividido em 2 pedaços (um por stream), cada um múltiplo de 1024 para
   não cortar um bloco: `CHUNK = ceil(nblocos / NSTREAMS) * THREADS`.
3. **Fase 1** (para cada stream `s`): a CPU **gera** o pedaço `s`, depois `cudaMemcpyAsync` H2D
   e `scan_blocos` na stream `s`. Enquanto a GPU copia e faz o scan do pedaço 0, a CPU já está
   gerando o pedaço 1 → **CPU e GPU trabalhando ao mesmo tempo**.
4. **Dependência entre pedaços**: o scan das somas precisa de todos os blocos prontos →
   `cudaStreamSynchronize` nas 2 streams, depois o scan das somas na stream 0 e
   `cudaStreamSynchronize(streams[0])`.
5. **Fase 2** (para cada stream `s`): `somar_offset` → `normalizar` → `cudaMemcpyAsync` D2H. A
   ordem dentro da stream garante a dependência "scan → normalização → cópia". A cópia do pedaço 0
   sobrepõe o cálculo do pedaço 1.
6. `cudaStreamSynchronize` em todas antes de a CPU ler o resultado.

```
CPU      | gera 0 | gera 1 |              espera                       |
stream 0 |        | H2D 0 → scan 0 |  sync  | scan somas | offset0 → norm0 → D2H 0 |
stream 1 |                 | H2D 1 → scan 1 |                 | offset1 → norm1 → D2H 1 |
```

## Passo 6 — Compilar e rodar no Franky

```bash
module load cuda/12.8.1
g++ -O3 base.cpp -o base
nvcc -O3 -arch=sm_86 sincrono.cu -o sincrono
nvcc -O3 -arch=sm_86 assincrono.cu -o assincrono
sbatch ex1.slurm        # depois: cat ex1.out
# ou direto: srun --partition=merry_gpu --gres=gpu:1 ./assincrono
```
`-arch=sm_86` é a arquitetura da RTX A1000 do Franky. Sem ele, o nvcc gera código genérico, e o
1º lançamento de kernel faz compilação JIT (~120 ms **dentro** do tempo medido).

## Passo 7 — Resultados (Franky, RTX A1000, 5 execuções)

| Versão | Tempo (mediana) |
|---|---|
| CPU (base) | 0,056 ms |
| GPU síncrona | 0,185 ms |
| GPU assíncrona | 0,177 ms (~4–7% melhor que a síncrona) |

Saída da GPU (igual nas duas versões):
```
[ 7.68173e+09 ... 7.68314e+09 7.68331e+09 ]
[ 0.999795 0.999817 0.99984 ... 0.999977 1 ]
```

**Por que a 5ª casa difere da CPU?** A soma exata é N·(N+1) = 7.683.311.370 = **7,68331e9**. A GPU
acerta. A CPU soma 87 mil `float` em sequência e acumula erro de arredondamento (dá 7,68322e9). A
GPU soma em árvore, com parcelas de tamanho parecido, e erra menos. Não é bug, é a ordem da soma em
ponto flutuante.

**O que explicar na entrega:**
- Com N = 87.654 (350 KB), a GPU é **mais lenta** que a CPU: o tempo é dominado por custos fixos
  (lançar kernels, cópias, sincronizações) e não pelo cálculo.
- A versão assíncrona ganha pouco porque há pouco a sobrepor. Com mais pedaços fica **pior**
  (testado: 6 pedaços = 0,215 ms, 11 pedaços = 0,26 ms), porque cada pedaço acrescenta lançamentos.
- A 1ª execução de um binário recém-compilado sai mais lenta (~2 ms, aquecimento; as seguintes
  ficam estáveis). Por isso o slurm repete 5 vezes e o valor reportado é a mediana.
