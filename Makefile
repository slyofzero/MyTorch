CXX = clang++
CXXFLAGS = -std=c++23 -Wall -Wextra -O3
INCLUDES = -Iinclude $(shell .venv/bin/python -m pybind11 --includes 2>/dev/null || python3 -m pybind11 --includes)
EXT_SUFFIX = $(shell python3-config --extension-suffix)

all: python

cpp:
	$(CXX) $(CXXFLAGS) -Iinclude src/tensor.cpp -o tensor
	./tensor

python:
	$(CXX) $(CXXFLAGS) -shared -fPIC $(INCLUDES) src/tensor.cpp src/bindings.cpp -o mytorch/_C$(EXT_SUFFIX)

test-python: python
	.venv/bin/python main.py

clean:
	rm -f tensor mytorch/*.so mytensor*.so
