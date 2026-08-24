CXX = clang++
CXXFLAGS = -std=c++23 -Wall -Wextra

FILE ?= src/tensor.cpp

all: run

run:
	$(CXX) $(CXXFLAGS) $(FILE) -o tensor
	./tensor

clean:
	rm -f tensor
