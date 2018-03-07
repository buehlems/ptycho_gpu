//from  https://docs.python.org/2/extending/extending.html
//      https://docs.python.org/2/extending/building.html#building
//and   https://docs.python.org/2/install/
//http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

#include <Python.h>
// #include "/usr/lib64/python2.6/site-packages/numpy/core/include/numpy/arrayobject.h"
#include <numpy/arrayobject.h>

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

    return PyArray_Return(pyX);
    
}

static PyMethodDef p2pMethods[] = {
    // ...
    { "py2petsc",  py2petsc, METH_VARARGS, "run petsc"	},
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

