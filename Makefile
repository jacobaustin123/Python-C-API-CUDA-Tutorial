all:
	nvcc -rdc=true --compiler-options '-fPIC' -c -o temp.o vectorAdd/vectorAdd.cu
	nvcc -dlink --compiler-options '-fPIC' -o vectorAdd.o temp.o -lcudart
	rm -f libvectoradd.a
	ar cru libvectoradd.a vectorAdd.o temp.o
	ranlib libvectoradd.a

test: all
	g++ tests/test.c -L. -lvectoradd -o main -L/usr/local/cuda/lib64 -lcudart

python: all
	CC=g++ LDSHARED='g++ -pthread -shared -B /home/jacob/anaconda3/compiler_compat -L/home/jacob/anaconda3/lib -Wl,-rpath=/home/jacob/anaconda3/lib -Wl,--no-as-needed -Wl,--sysroot=/' python setup.py build
	python setup.py install

clean:
	rm -f libvectoradd.a *.o main
	rm -rf build
