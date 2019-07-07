#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

extern double * myVectorAdd(double * h_A, double * h_B, int numElements);

static PyObject* helloworld(PyObject* self, PyObject* args) {
    printf("Hello World\n");
    return Py_None;
}

static unsigned long long * memo = NULL;

unsigned long long Cfib(int n) {
    unsigned long long value;

    if (n < 2 || memo[n] != 0) 
        return memo[n];
    else {
        value = Cfib(n-1) + Cfib(n-2);
        memo[n] = value;
        return value;
    }
}

// Our Python binding to our C function
// This will take one and only one non-keyword argument
static PyObject* fib(PyObject* self, PyObject* args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n))
        return NULL;
    
    if (n < 2) {
        return Py_BuildValue("i", n);
    }

    memo = (unsigned long long *) calloc(n + 1, sizeof(unsigned long long));  // memoization
    if (memo == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Unable to dynamically allocate memory for memoization.");
        return NULL;
    }

    memo[0] = 0; // set initial conditions
    memo[1] = 1;
    
    // return our computed fib number
    PyObject* value = PyLong_FromUnsignedLongLong(Cfib(n));
    free(memo);
    return Py_BuildValue("N", value);
}

static PyObject* vector_add(PyObject* self, PyObject* args) {
    PyArrayObject *array1, *array2;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    PyArrayObject* array3 = (PyArrayObject*)PyArray_NewLikeArray(array1, NPY_ANYORDER, NULL, 1);

    PyObject * mobj = PyArray_MultiIterNew(3, array1, array2, array3);
    if (!mobj) {
        return NULL;
    }

    double *data1, *data2, *data3;

    while (PyArray_MultiIter_NOTDONE(mobj)) {
        data1 = (double *) PyArray_MultiIter_DATA(mobj, 0);
        data2 = (double*) PyArray_MultiIter_DATA(mobj, 1);
        data3 = (double*) PyArray_MultiIter_DATA(mobj, 2);

        *data3 = *data1 + *data2;

        PyArray_MultiIter_NEXT(mobj);
    }

    // array1 = (PyArrayObject*) PyArray_ContiguousFromObject(array1, PyArray_DOUBLE, 1, 3); // range of allowed dimensions
    // array2 = (PyArrayObject*) PyArray_ContiguousFromObject(array2, PyArray_DOUBLE, 1, 3);

    // if (array1 == NULL || array2 == NULL) {
    //     PyErr_SetString(PyExc_RuntimeError, "arrays cannot be converted to double type or have invalid size."); 
    //     return NULL;
    // }

    // if (PyArray_NDIM(array1) != PyArray_NDIM(array2)) {
    //     PyErr_SetString(PyExc_RuntimeError, "arrays must have the same number of dimensions.");
    //     return NULL;
    // }

    // for (int i = 0; i < array1 -> nd; i++) {
    //     if (PyArray_DIMS(array1)[i] != PyArray_DIMS(array2)[i]) {
    //         PyErr_SetString(PyExc_RuntimeError, "arrays must have the same shape.");
    //         return NULL;
    //     }
    // }

    // PyArrayObject* output = (PyArrayObject*) PyArray_NewLikeArray(array1, NPY_ANYORDER, NULL, 1);

    // PyArrayIterObject* iter1, * iter2, * iter3;
    // iter1 = (PyArrayIterObject*) PyArray_IterNew(array1);
    // iter2 = (PyArrayIterObject*) PyArray_IterNew(array2);
    // iter3 = (PyArrayIterObject*) PyArray_IterNew(output);

    // if (iter1 == NULL || iter2 == NULL || iter3 == NULL) {
    //     PyErr_SetString(PyExc_RuntimeError, "unable to create iterator.");
    //     return NULL;
    // }

    // while (iter1->index < iter1->size) {
    //     *(double*)(iter3 -> dataptr) = * (double*) (iter1 -> dataptr) + * (double*) (iter2->dataptr);

    //     PyArray_ITER_NEXT(iter1);
    //     PyArray_ITER_NEXT(iter2);
    //     PyArray_ITER_NEXT(iter3);
    // }

    Py_DECREF(array1);
    Py_DECREF(array2);

    return (PyObject *) array3;
}

static PyMethodDef myMethods[] = {
    {"vector_add", vector_add, METH_VARARGS, "add two 1D numpy arrays together"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT, "myModule",
    "myModule", -1, myMethods
};

PyMODINIT_FUNC PyInit_myModule(void) {
    import_array(); 
    return PyModule_Create(&myModule);
}

