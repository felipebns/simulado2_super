# Exercício 3 — Scatter/Gather → ponto a ponto não bloqueante (3 pts)

Arquivos: `resolucao/ex3_mpi/` → `base.cpp` (enunciado), `nao_bloqueante.cpp`, `ex3.slurm`.

Para os 3,0 pontos é preciso: `MPI_Isend`/`MPI_Irecv` (não pode `MPI_Send`/`MPI_Recv`), **com**
`MPI_Wait`/`MPI_Waitall` antes de usar os buffers, e a soma certa (1 + 2 + … + 16 = **136**).

---

## Passo 1 — O que cada coletiva fazia

| Coletiva | Quem faz o quê | Troca por |
|---|---|---|
| `MPI_Scatter` | rank 0 manda o bloco `p` para o rank `p` (o bloco 0 fica com ele) | rank 0: `Isend` para p = 1..size-1 + copia o bloco 0; outros: `Irecv` |
| `MPI_Gather` | cada rank manda 1 int; rank 0 guarda em `somas[p]` | rank 0: `Irecv` de p = 1..size-1 + `somas[0] = soma_local`; outros: `Isend` |

## Passo 2 — Distribuição (no lugar do `MPI_Scatter`)

```cpp
std::vector<MPI_Request> pedidos(size);      // um pedido por mensagem pendente

if (rank == 0) {
    for (int p = 1; p < size; p++)
        MPI_Isend(dados.data() + p * tam_local, tam_local, MPI_INT,
                  p, TAG_DADOS, MPI_COMM_WORLD, &pedidos[p]);
    for (int i = 0; i < tam_local; i++) local[i] = dados[i];   // bloco 0 é dele
} else {
    MPI_Irecv(local.data(), tam_local, MPI_INT, 0, TAG_DADOS, MPI_COMM_WORLD, &pedidos[0]);
    MPI_Wait(&pedidos[0], MPI_STATUS_IGNORE);   // precisa dos dados antes de somar
}

int soma_local = 0;                             // rank 0 soma enquanto os envios andam
for (int x : local) soma_local += x;

if (rank == 0)                                  // envios terminaram antes de reutilizar pedidos
    MPI_Waitall(size - 1, pedidos.data() + 1, MPI_STATUSES_IGNORE);
```
- O deslocamento `p * tam_local` é o mesmo que o Scatter usaria.
- O rank 0 **não** manda mensagem para si mesmo: copia o próprio bloco.
- O `MPI_Waitall` do rank 0 vem **depois** da soma local, então a soma dele acontece enquanto as
  mensagens estão sendo enviadas (sobreposição, como no exemplo `nao_bloqueante.cpp` da aula).
  Isso só é seguro porque `dados` não é alterado enquanto os envios estão pendentes.

## Passo 3 — Coleta (no lugar do `MPI_Gather`)

```cpp
if (rank == 0) {
    somas[0] = soma_local;
    for (int p = 1; p < size; p++)
        MPI_Irecv(&somas[p], 1, MPI_INT, p, TAG_SOMA, MPI_COMM_WORLD, &pedidos[p]);
    MPI_Waitall(size - 1, pedidos.data() + 1, MPI_STATUSES_IGNORE);  // só lê depois disso
} else {
    MPI_Isend(&soma_local, 1, MPI_INT, 0, TAG_SOMA, MPI_COMM_WORLD, &pedidos[0]);
    MPI_Wait(&pedidos[0], MPI_STATUS_IGNORE);   // soma_local válida até o envio terminar
}
```
- Cada `Irecv` grava em `somas[p]`, então a ordem de chegada não importa.
- Tags diferentes para dados (10) e somas (20) evitam confundir as mensagens.
- O resto (somar `somas` e imprimir) é igual à base.

**Regras dos buffers (aula "Comunicação Bloqueante e Não Bloqueante"):** enquanto o pedido está
pendente, não altere o buffer de envio e não leia o de recebimento. Sem `Wait` = 1,5 pt.

## Passo 4 — Compilar e rodar

```bash
mpic++ -O2 base.cpp -o base
mpic++ -O2 nao_bloqueante.cpp -o nao_bloqueante
sbatch ex3.slurm                 # 4 processos em 2 nós; depois: cat ex3.out
# ou: srun --partition=merry_cpu --mpi=pmix --mem=1G -n 4 ./nao_bloqueante
```
**Atenção:** sem `--mem=1G` o `srun` fica preso na fila com `(MaxMemPerLimit)` (pede mais memória
que o limite da partição).

## Passo 5 — Resultado

```
========= Coletivo (base) =========
Soma total = 136
========= Ponto a ponto não bloqueante =========
Soma total = 136
```
Testado também com `-n 1, 2, 8, 16`: sempre 136. Assim como a base, assume que `N` é divisível
pelo número de processos.
