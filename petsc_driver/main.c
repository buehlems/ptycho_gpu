

#include <mpi.h>
#include <stdio.h>
#include <petscsys.h>

#include "ptycho_petsc.h"


int main(int argc, char** argv) {

	int npes;
	int rank;

// Initialize the MPI environment

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Get the name of the processor

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

// Print off a hello world message
	printf("petsc_driver on %s, rank %d, Number of processes %d\n", processor_name, rank, npes);

// petsc

	ptycho_setup_petsc(argc, argv);

	ptycho_read_and_fill_Matrix();

	ptycho_read_and_set_RHS();

	ptycho_petsc_solve();

	ptycho_petsc_get_solution ();

// Finalize the MPI environment.
	MPI_Finalize();
}
