//from  https://docs.python.org/2/extending/extending.html
//      https://docs.python.org/2/extending/building.html#building
//and   https://docs.python.org/2/install/
//http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

#include <Python.h>
// #include "/usr/lib64/python2.6/site-packages/numpy/core/include/numpy/arrayobject.h"
#include <numpy/arrayobject.h>

int mpiHelloWorld(void);

/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}
/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
	free((char*) v);
}


int  not_doublevector(PyArrayObject *vec)
{

    if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)
    {

	PyErr_SetString(PyExc_ValueError,
			"Array must be of type Float and 1 dimensional (n).");

	return 1;

    }


    return 0;

}

int  not_intvector(PyArrayObject *vec)
{

    if (vec->descr->type_num != NPY_INT || vec->nd != 1)
    {
	PyErr_SetString(PyExc_ValueError,
			"Array must be of type int and 1 dimensional (n).");
	
	return 1;

    }


    return 0;

}

/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
	if (mat->descr->type_num != NPY_DOUBLE || mat->nd != 2)  {
		PyErr_SetString(PyExc_ValueError,
			"In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
		return 1;  }
	return 0;
}


/* ==== Create 1D Carray from PyArray ======================
   Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  
{

    // int i,n;


    // int n=arrayin->dimensions[0];

    return (double *) arrayin->data;
    /* pointer to arrayin data as double */
}

/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i,n,m;
	
	n=arrayin->dimensions[0];
	m=arrayin->dimensions[1];
	c=ptrvector(n);
	a=(double *) arrayin->data;  /* pointer to arrayin data as double */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}

static PyObject *printNPY(PyObject *self, PyObject *args)
{
    printf("NPY_BOOL=%d\n",NPY_BOOL);
    printf("NPY_BYTE=%d\n",NPY_BYTE);  
    printf("NPY_INT8=%d\n",NPY_INT8);
    printf("NPY_SHORT=%d\n",NPY_SHORT);
    printf("NPY_INT16=%d\n",NPY_INT16);
    printf("NPY_INT=%d\n",NPY_INT);
    printf("NPY_INT32=%d\n",NPY_INT32);
    printf("NPY_LONG=%d\n",NPY_LONG);
    printf("NPY_LONGLONG=%d\n",NPY_LONGLONG);
    printf("NPY_INT64=%d\n",NPY_INT64);
    printf("NPY_UBYTE=%d\n",NPY_UBYTE);
    printf("NPY_UINT8=%d\n",NPY_UINT8);
    printf("NPY_USHORT=%d\n",NPY_USHORT);
    printf("NPY_UINT16=%d\n",NPY_UINT16);
    printf("NPY_UINT=%d\n",NPY_UINT);
    printf("NPY_UINT32=%d\n",NPY_UINT32);
    printf("NPY_ULONG=%d\n",NPY_ULONG);
    printf("NPY_ULONGLONG=%d\n",NPY_ULONGLONG);
    printf("NPY_UINT64=%d\n",NPY_UINT64);
    // printf("NPY_HALF=%d\n",NPY_HALF);
    // printf("NPY_FLOAT16=%d\n",NPY_FLOAT16);
    printf("NPY_FLOAT=%d\n",NPY_FLOAT);
    printf("NPY_FLOAT32=%d\n",NPY_FLOAT32);
    printf("NPY_DOUBLE=%d\n",NPY_DOUBLE);
    printf("NPY_FLOAT64=%d\n",NPY_FLOAT64);
    printf("NPY_LONGDOUBLE=%d\n",NPY_LONGDOUBLE);
    printf("NPY_CFLOAT=%d\n",NPY_CFLOAT);
    printf("NPY_COMPLEX64=%d\n",NPY_COMPLEX64);
    printf("NPY_CDOUBLE=%d\n",NPY_CDOUBLE);
    printf("NPY_COMPLEX128=%d\n",NPY_COMPLEX128);
    printf("NPY_CLONGDOUBLE=%d\n",NPY_CLONGDOUBLE);
    // printf("NPY_DATETIME=%d\n",NPY_DATETIME);
    // printf("NPY_TIMEDELTA=%d\n",NPY_TIMEDELTA);
    printf("NPY_STRING=%d\n",NPY_STRING);
    printf("NPY_UNICODE=%d\n",NPY_UNICODE);
    printf("NPY_OBJECT=%d\n",NPY_OBJECT);
    printf("NPY_VOID=%d\n",NPY_VOID);
    printf("NPY_INTP=%d\n",NPY_INTP);
    printf("NPY_UINTP=%d\n",NPY_UINTP);
    // printf("NPY_MASK=%d\n",NPY_MASK);
    printf("NPY_DEFAULT_TYPE=%d\n",NPY_DEFAULT_TYPE);
    printf("NPY_NTYPES=%d\n",NPY_NTYPES);
    printf("NPY_NOTYPE=%d\n",NPY_NOTYPE);
    printf("NPY_USERDEF=%d\n",NPY_USERDEF       );

    printf("\n");
    
    printf("NPY_BOOLLTR=%c\n",    NPY_BOOLLTR);
    printf("NPY_BYTELTR=%c\n",NPY_BYTELTR);
    printf("NPY_UBYTELTR=%c\n",	NPY_UBYTELTR);
    printf("NPY_SHORTLTR=%c\n",NPY_SHORTLTR);
    printf("NPY_USHORTLTR=%c\n",NPY_USHORTLTR);
    printf("NPY_INTLTR=%c\n",NPY_INTLTR);
    printf("NPY_UINTLTR=%c\n",NPY_UINTLTR);
    printf("NPY_LONGLTR=%c\n",NPY_LONGLTR);
    printf("NPY_ULONGLTR=%c\n",NPY_ULONGLTR);
    printf("NPY_LONGLONGLTR=%c\n",NPY_LONGLONGLTR);
    printf("NPY_ULONGLONGLTR=%c\n",NPY_ULONGLONGLTR);
    // printf("NPY_HALFLTR=%c\n",NPY_HALFLTR);
    printf("NPY_FLOATLTR=%c\n",NPY_FLOATLTR);
    printf("NPY_DOUBLELTR=%c\n",NPY_DOUBLELTR);
    printf("NPY_LONGDOUBLELTR=%c\n",NPY_LONGDOUBLELTR);
    printf("NPY_CFLOATLTR=%c\n",NPY_CFLOATLTR);
    printf("NPY_CDOUBLELTR=%c\n",NPY_CDOUBLELTR);
    printf("NPY_CLONGDOUBLELTR=%c\n",NPY_CLONGDOUBLELTR);
    printf("NPY_DATETIMELTR=%c\n",NPY_DATETIMELTR);
    printf("NPY_TIMEDELTALTR=%c\n",NPY_TIMEDELTALTR);
    printf("NPY_OBJECTLTR=%c\n",NPY_OBJECTLTR);
    printf("NPY_STRINGLTR=%c\n",NPY_STRINGLTR);
    printf("NPY_VOIDLTR=%c\n",NPY_VOIDLTR);
    printf("NPY_INTPLTR=%c\n",NPY_INTPLTR);
    printf("NPY_UINTPLTR=%c\n",NPY_UINTPLTR);
    printf("NPY_GENBOOLLTR=%c\n",NPY_GENBOOLLTR);
    printf("NPY_SIGNEDLTR=%c\n",NPY_SIGNEDLTR);
    printf("NPY_UNSIGNEDLTR=%c\n",NPY_UNSIGNEDLTR);
    printf("NPY_FLOATINGLTR=%c\n",NPY_FLOATINGLTR);
    printf("NPY_COMPLEXLTR=%c\n",NPY_COMPLEXLTR);
    return Py_BuildValue("i", 0);
    
}


// run with slrum:
// salloc --nodes=1 --tasks-per-node=1  --time=00:05:00
// srun python create_test_matrices.py
static PyObject *py2petsc(PyObject *self, PyObject *args)
{

    PyArrayObject *pyAdata = NULL;
    PyArrayObject *pyAindex = NULL;
    PyArrayObject *pyAiptr = NULL;
    PyArrayObject *pyB = NULL;
    PyArrayObject *pyX = NULL;


    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &pyAdata, &PyArray_Type, &pyAindex, &PyArray_Type, &pyAiptr, &PyArray_Type, &pyB)) 
	return NULL;

    // convert py arrays to C arrays
    // Adata
    printf("create Adata\n");
    
    double *Adata;
    int nAdata;
    npy_intp dimsAdata[2];
    if (not_doublevector(pyAdata)){
	printf("pyAdata is not a double vector. Exit\n");
	return NULL;
    }
    /* Change contiguous arrays into C * arrays   */
    Adata =pyvector_to_Carrayptrs(pyAdata);
    /* Get vector dimension. */
    nAdata=pyAdata->dimensions[0];
    dimsAdata[0]=nAdata;

    // Aindex
    printf("create Aindex\n");
    int *Aindex;
    int nAindex;
    npy_intp dimsAindex[2];
    if (not_intvector(pyAindex)){
	printf("pyAindex is not a int vector. Exit\n");
	return NULL;
    }
    /* Change contiguous arrays into C * arrays   */
    Aindex =(int *)pyvector_to_Carrayptrs(pyAindex);
    /* Get vector dimension. */
    nAindex=pyAindex->dimensions[0];
    dimsAindex[0]=nAindex;

    // Aiptr
    printf("create Aiptr\n");

    int *Aiptr;
    int nAiptr;
    npy_intp dimsAiptr[2];
    if (not_intvector(pyAiptr)){
	printf("pyAiptr is not a int vector. Exit\n");
	return NULL;
    }
    /* Change contiguous arrays into C * arrays   */
    Aiptr =(int *)pyvector_to_Carrayptrs(pyAiptr);
    /* Get vector dimension. */
    nAiptr=pyAiptr->dimensions[0];
    dimsAiptr[0]=nAiptr;

    // B
    printf("create B\n");

    double **B;
    int nB,mB;
    if (not_doublematrix(pyB)){
	printf("pyB is not a double matrix. Exit\n");
	return NULL;
    }
    /* Change contiguous arrays into C * arrays   */
    B=pymatrix_to_Carrayptrs(pyB);
    /* Get vector dimension. */
    nB=pyB->dimensions[0];
    mB=pyB->dimensions[1];

    // result X	
    printf("create X\n");
    double **X;
    npy_intp dimsX[2];
    int nX,mX;
    
    // this is probably not right yet
    dimsX[0]=nB;
    dimsX[1]=mB;

    pyX=(PyArrayObject *) PyArray_SimpleNew(2,dimsX,NPY_DOUBLE);
    X =pymatrix_to_Carrayptrs(pyX);
    /* Get vector dimension. */
    nX=pyX->dimensions[0];
    mX=pyX->dimensions[1];
    

    
    // printf("vector size=%d\n",n);
    // print arrays
    int i,j;

    printf("Adata\n");
    printf("n=%d\n",nAdata);
    for(i=0; i<nAdata; i++)
    {
	printf("%2.2f ",Adata[i]);
    }
    printf("\n");

    printf("Aindex\n");
    printf("n=%d\n",nAindex);
    for(j=0; j<nAindex; j++)
    {
	printf("%d ",Aindex[j]);
    }
    printf("\n");

    printf("Aiptr\n");
    printf("n=%d\n",nAiptr);
    for(j=0; j<nAiptr; j++)
    {
	printf("%d ",Aiptr[j]);
    }
    printf("\n");

    printf("B\n");
    printf("n=%d m=%d\n",nB,mB);
    for(i=0; i<nB; i++)
    {
	for(j=0; j<mB; j++)
	{
	    printf("%2.2f ",B[i][j]);
	}
	printf("\n");
    }
    printf("\n");

    // call petSC to calc X
    // copy result from X to pyX
    free_Carrayptrs(B);
    free_Carrayptrs(X);

    mpiHelloWorld();

    return PyArray_Return(pyX);
}

// MPI part
#include <mpi.h>
#include <stdio.h>
#include <petscsys.h>

#include "ptycho_petsc.h"


int mpiHelloWorld() {

// Initialize the MPI environment

	MPI_Init(NULL, NULL);

	int npes;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// Get the name of the processor

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

// Print off a hello world message
	printf("petsc_driver on %s, rank %d, Number of processes %d\n", processor_name, rank, npes);

/* // petsc */

	// ptycho_setup_petsc(argc, argv);
	ptycho_setup_petsc(NULL, NULL);

	ptycho_read_and_fill_Matrix();

	ptycho_read_and_set_RHS();

	ptycho_petsc_solve();

	ptycho_petsc_get_solution ();

// Finalize the MPI environment.
	MPI_Finalize();
	return(0);
}



static PyMethodDef p2pMethods[] = {
    // ...
    { "py2petsc",  py2petsc, METH_VARARGS, "run petsc"	},
    { "printNPY",  printNPY, METH_VARARGS, "print the NPY values"	},
    //		,
	//    ...
	{ NULL, NULL, 0, NULL	}
	/* Sentinel */
};

PyMODINIT_FUNC
initp2p(void)
{

    (void) Py_InitModule("p2p", p2pMethods);
    import_array();  // Must be present for NumPy.  Called first after above line.

}

int
main(int argc, char *argv[])
{

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);


    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();


    /* Add a static module */
    initp2p();

    printf("Initp2p done\n");
    return 1;

}

