
// function prototypes

void ptycho_setup_petsc (int M,int N);
void ptycho_read_and_fill_Matrix (PetscScalar * values,int * indices,int * row_pointer,int rows);
void ptycho_read_and_set_RHS (PetscScalar ** B);
void ptycho_petsc_solve (void);
void ptycho_petsc_get_solution (PetscScalar ** X);
