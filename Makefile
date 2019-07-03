.PHONY: all build test clean

all: build
	CC=g++ LDSHARED='$(shell python scripts/configure.py)' python setup.py build
	python setup.py install
	python tests/test.py

build:
	nvcc -rdc=true --compiler-options '-fPIC' -c -o temp.o vectorAdd/vectorAdd.cu
	nvcc -dlink --compiler-options '-fPIC' -o vectorAdd.o temp.o -lcudart
	rm -f libvectoradd.a
	ar cru libvectoradd.a vectorAdd.o temp.o
	ranlib libvectoradd.a

test: build
	g++ tests/test.c -L. -lvectoradd -o main -L${CUDA_PATH}/lib64 -lcudart

clean:
	rm -f libvectoradd.a *.o main temp.py
	rm -rf build
