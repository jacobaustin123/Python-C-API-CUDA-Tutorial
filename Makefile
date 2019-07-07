.PHONY: all clean

all:
	python3.7 setup.py build
	python3.7 setup.py install

clean:
	rm -rf build
