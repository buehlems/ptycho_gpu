
#include <stdlib.h>
#include <stdio.h>

#include <unistd.h>
#include <fcntl.h>

#include <mpi.h>

#include <petscsys.h>
#include <petscmat.h>
#include <petscksp.h>
#include "petscksp.h"

#include <stdlib.h>
//#include "petscmat.h"

int   ROWS,COLS;
Mat   A;
KSP   ksp;          /* linear solver context */
PC    precond;
Vec   rhs;
Vec   sol;

void ptycho_setup_petsc(int rows,int cols) {
    ROWS=rows;
    COLS=cols;
	printf("here is init_petsc\n");

	printf ("matrix size %d %d \n",rows,cols);
	PetscInitialize(NULL, NULL, NULL, NULL);

// Setup Matrix

	MatCreate(PETSC_COMM_WORLD,&A);
	MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,rows, cols);
	MatSetFromOptions(A);
	MatSetUp(A);

// Setup rhs and solution vector

   VecCreateMPI (PETSC_COMM_WORLD, PETSC_DECIDE, rows, &rhs);
   VecCreateMPI (PETSC_COMM_WORLD, PETSC_DECIDE, cols, &sol);

// Setup ksp

	KSPCreate(PETSC_COMM_WORLD,&ksp);

// Setup Preconditioner

	PCCreate (PETSC_COMM_WORLD, &precond);

	return;
};

void ptycho_read_and_fill_Matrix (PetscScalar * values,int * indices,int * row_pointer,int rows) {

	int    i,irow;
	int    nval;
	size_t max_size = ROWS*sizeof(int);
	/* int    *row_pointer;
	int    *indices;
	PetscScalar *values;
     

	row_pointer = malloc(max_size);
	indices     = malloc(max_size);
	values      = malloc(max_size*4);
     
	int fd = open("data/a_indpointer.bin", O_RDONLY);
	if(fd < 0)  {
		printf("file not found:  a_indpointer.bin\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	size_t bytes_read = read(fd, row_pointer, max_size);

	int rows = bytes_read/sizeof(int);
	printf ("ind_pointer bytes read %ld %d %ld %ld\n",bytes_read,rows,max_size,sizeof(PetscScalar));

	int fd_ind = open("data/a_ind.bin", O_RDONLY);
	if(fd_ind < 0)  {
		printf("file not found:  a_ind.bin\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	int fd_dat = open("data/a_data.bin", O_RDONLY);
	if(fd_dat < 0)  {
		printf("file not found:  a_data.bin\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}*/
    printf("rows %d\n",rows);
   for (irow=0; irow<rows; irow++ ) {
   		nval = row_pointer[irow+1] - row_pointer[irow];

   	//size_t ind_nval = read (fd_ind, indices, nval*sizeof(int));
   //	size_t val_nval = read (fd_dat, values, nval*sizeof(double complex));
   	//MatSetValues (A, 1, &irow, nval, &indices[irow], &values[irow], INSERT_VALUES);
       int row_start = row_pointer[irow];
       MatSetValues (A, 1, &irow, nval, &indices[row_start], &values[row_start], INSERT_VALUES);
   	/*if(irow < 10) printf ("rows %d %d %d %d %d\n",irow,row_pointer[irow],nval,nval,row_pointer[irow]);
   	if(irow < 10) printf ("row indices %d %d %d %f + i%f\n",irow,nval,nval,creal(values[irow]),cimag(values[irow]));
   */
   }
   MatAssemblyBegin (A, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd   (A, MAT_FINAL_ASSEMBLY);
    
    printf("Matrix A as generated in PETSc\n");
    MatView(A,PETSC_VIEWER_STDOUT_SELF);
    
	return;
}

void ptycho_read_and_set_RHS (double ** B) {

	int    i;
	size_t max_size = ROWS*sizeof(double);
	double *buf;
	int    *indices;
	PetscScalar *rhs_val;
    printf("rows=%d\n",ROWS);
	//buf     = malloc(max_size);
	rhs_val= malloc(ROWS*sizeof(PetscScalar));

	/* int fd = open("data/b_vector.bin", O_RDONLY);
	if(fd < 0)  {
		printf("file not found:  b_vector.bin\n");
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	size_t bytes_read = read(fd, buf, max_size*sizeof(double));

	int rows = bytes_read/sizeof(double);
	printf ("RHS bytes read %ld %d %ld\n",bytes_read,rows,max_size*sizeof(double));
	printf ("RHS values %f %f %f\n",buf[0],buf[1],buf[2]);
     */
	indices = malloc(ROWS*sizeof(int));

	for (i=0; i<ROWS; i++) {
		indices[i] = i;
		rhs_val[i] = (PetscScalar) B[1][i];          // convert double to double complex //use first only one column of B (vector case)
        printf("B[%d][1]=%lf %lf\n",i,B[i][1],B[i+1][1]);
	}
   VecSetValues (rhs, ROWS, indices, rhs_val, INSERT_VALUES);

   VecAssemblyBegin (rhs);
   VecAssemblyEnd   (rhs);

    printf("Vector RHS as generated in PETSc\n");
    VecView(rhs,PETSC_VIEWER_STDOUT_SELF);
	return;

}

void ptycho_petsc_solve (void) {

   enum solver_type {
   	GMRES,
		BICG,
		BCGS,
		FGMRES,
		DGMRES,
		GCR,
		RICHARDSON,
		CHEBYSHEV
   };

   enum preconditioner_type {
   	NOPC,
		JACOBI,
		SOR,
		EISENSTAT,
		ICC,
		ILU
   };

// select solver and preconditioner

	int solver_selected = BICG;
	int pc_selected     = NOPC;

	int gmres_max_restart_iter = 20;

   KSPSetOperators(ksp, A, A);

   KSPSetInitialGuessNonzero (ksp, PETSC_FALSE);


   switch(solver_selected) {
   	case GMRES:
   		KSPSetType (ksp, KSPGMRES);
   		if(gmres_max_restart_iter > 20) KSPGMRESSetRestart (ksp, gmres_max_restart_iter);
   		break;
   	case BICG:
   		KSPSetType (ksp, KSPBICG);
   		break;
   	case BCGS:
   		KSPSetType (ksp, KSPBCGS);
   		break;
   	case FGMRES:
   		KSPSetType (ksp, KSPFGMRES);
   		break;
   	case DGMRES:
   		KSPSetType (ksp, KSPDGMRES);
   		break;
   	case GCR:
   		KSPSetType (ksp, KSPGCR);
   		break;
   	case RICHARDSON:
   		KSPSetType (ksp, KSPRICHARDSON);
   		break;
   	case CHEBYSHEV:
   		KSPSetType (ksp, KSPCHEBYSHEV);
   		break;
   	default:
   		printf("illegal solver value %d\n",solver_selected);
   };

   KSPType solver_type_as_string;
   KSPGetType (ksp, &solver_type_as_string);
   printf("solver type %s\n",solver_type_as_string);

   KSPGetPC (ksp, &precond);

   switch(pc_selected) {
   	case NOPC:
   		PCSetType (precond, PCNONE);
   		break;
   	case JACOBI:
   		PCSetType (precond, PCJACOBI);
   		break;
   	case SOR:
   		PCSetType (precond, PCSOR);
   		break;
   	case EISENSTAT:
   		PCSetType (precond, PCEISENSTAT);
   		break;
   	case ICC:
   		PCSetType (precond, PCICC);
   		break;
   	case ILU:
   		PCSetType (precond, PCILU);
   		break;
   	default:
   		printf("illegal preconditioner value %d\n",solver_selected);
   };

   KSPType pc_type_as_string;
   PCGetType (precond, &pc_type_as_string);
   printf("preconditioner type %s\n",pc_type_as_string);

   PetscReal rtol;
   PetscReal abstol;
   PetscReal dtol;
   PetscInt  maxits;

   KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &maxits);
   printf ("Default tolerances %f %f %f %d\n", rtol, abstol, dtol, maxits);

// Here: modify Tolerances

   KSPSetTolerances (ksp, rtol, abstol, dtol, maxits);

   KSPSolve (ksp, rhs, sol);

   PetscInt niter = 0;
   KSPGetIterationNumber(ksp, &niter);
   printf ("Number of Iterations %d\n",niter);

	return;
};

void ptycho_petsc_get_solution (PetscScalar ** X) {
	int   i;

   // PetscScalar *x =X[1]; //use first only one column of X (vector case)


   KSPGetSolution (ksp, &sol);

   VecGetArray(sol, &X[1]);

   for (i=0; i<MIN(COLS,30); i++) {
   	printf("solution value %d %f + i%f \n",i,creal(X[1][i]),cimag(X[1][i]));
   //printf("solution value %d %f + i%f \n",i,creal(x[i]),cimag(x[i]));
   }

}

