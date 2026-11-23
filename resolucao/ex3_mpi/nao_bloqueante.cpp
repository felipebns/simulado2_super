#include <mpi.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int TAG_DADOS = 10;
    const int TAG_SOMA = 20;

    int N = 16;
    std::vector<int> dados;
    if (rank == 0) {
        dados.resize(N);
        for (int i = 0; i < N; i++) dados[i] = i+1;
    }

    int tam_local = N / size;
    std::vector<int> local(tam_local);

    // Um pedido por mensagem pendente (no rank 0, pedidos[p] é a mensagem do rank p)
    std::vector<MPI_Request> pedidos(size);

    // Distribuição dos dados (no lugar do MPI_Scatter)
    if (rank == 0) {
        for (int p = 1; p < size; p++)
            MPI_Isend(dados.data() + p * tam_local, tam_local, MPI_INT,
                      p, TAG_DADOS, MPI_COMM_WORLD, &pedidos[p]);
        // O bloco 0 fica com o próprio rank 0
        for (int i = 0; i < tam_local; i++) local[i] = dados[i];
    } else {
        MPI_Irecv(local.data(), tam_local, MPI_INT,
                  0, TAG_DADOS, MPI_COMM_WORLD, &pedidos[0]);
        // Precisa dos dados antes de somar
        MPI_Wait(&pedidos[0], MPI_STATUS_IGNORE);
    }

    // O rank 0 soma seu bloco enquanto os envios estão em andamento
    int soma_local = 0;
    for (int x : local) soma_local += x;

    // Garante que todos os envios terminaram antes de reutilizar os pedidos
    if (rank == 0)
        MPI_Waitall(size - 1, pedidos.data() + 1, MPI_STATUSES_IGNORE);

    std::vector<int> somas;
    if (rank == 0) somas.resize(size);

    // Coleta das somas locais (no lugar do MPI_Gather)
    if (rank == 0) {
        somas[0] = soma_local;
        for (int p = 1; p < size; p++)
            MPI_Irecv(&somas[p], 1, MPI_INT,
                      p, TAG_SOMA, MPI_COMM_WORLD, &pedidos[p]);
        // Só lê as somas depois que todas chegaram
        MPI_Waitall(size - 1, pedidos.data() + 1, MPI_STATUSES_IGNORE);
    } else {
        MPI_Isend(&soma_local, 1, MPI_INT,
                  0, TAG_SOMA, MPI_COMM_WORLD, &pedidos[0]);
        // Mantém soma_local válida até o envio terminar
        MPI_Wait(&pedidos[0], MPI_STATUS_IGNORE);
    }

    if (rank == 0) {
        int soma_total = 0;
        for (int s : somas) soma_total += s;
        std::cout << "Soma total = " << soma_total << "\n";
    }

    MPI_Finalize();
}
