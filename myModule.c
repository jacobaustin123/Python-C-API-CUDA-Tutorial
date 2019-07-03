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
    PyArrayObject* array1, * array2;
    double * output;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    if (array1 -> nd != 1 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }

    output = myVectorAdd((double *) array1 -> data, (double *) array2 -> data, n1);

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyObject* vector_add_cpu(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1, &PyArray_Type, &array2))
        return NULL;

    if (array1 -> nd != 1 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    printf("running vector_add on dim1: %d, stride1: %d, dim2: %d, stride2: %d\n", n1, array1->strides[0], n2, array2->strides[0]);

    if (n1 != n2) {
        PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
        return NULL;
    }

    double * output = (double *) malloc(sizeof(double) * n1);

    for (int i = 0; i < n1; i++)
        output[i] = *((double *) array1 -> data + i) + *((double *) array2 -> data + i);

    return PyArray_SimpleNewFromData(1, PyArray_DIMS(array1), PyArray_TYPE(array1), output);
}

static PyObject* trace(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    double sum;
    int i, n;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
        return NULL;

    if (array -> nd != 2 || array->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
        return NULL;
    }

    n = array->dimensions[0];
    if (n > array->dimensions[1])
        n = array->dimensions[1];
    
    sum = 0.0;

    for (i = 0; i < n; i++)
        sum += *(double*)(array->data + i * array->strides[0] + i * array->strides[1]);

    return PyFloat_FromDouble(sum);
}

static PyMethodDef myMethods[] = {
    {"helloworld", helloworld, METH_NOARGS, "Prints Hello World"},
    {"fib", fib, METH_VARARGS, "Computes the nth Fibonacci number"},
    {"trace", trace, METH_VARARGS, "Computes the trace of a 2-dimensional numpy float array"},
    {"vector_add_cpu", vector_add_cpu, METH_VARARGS, "add two vectors on the CPU"},
    {"vector_add", vector_add, METH_VARARGS, "add two vectors with CUDA"},
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

