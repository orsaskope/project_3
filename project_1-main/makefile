OBJS = search.o parameters.o LSH.o IVFFLAT.o ivfpq.o implementation.o hypercube.o
HEADERS = params.hpp LSH.h ivfflat.hpp ivfpq.hpp implementation.hpp hypercube.hpp
SOURCE = search.cpp parameters.cpp LSH.cpp IVFFLAT.cpp IVFPQ.cpp implementation.cpp hypercube.cpp
EXEC = search

ARGS = -ivfflat -type sift -w 150.0 -L 10 -k 2 -R 250 -d input.dat -q query.dat  -o output.txt -range true -M 3000 -probes 300 -kproj 18 -N 5 -kclusters 32  -nprobe 2 -seed 9 -nbits 6

CC =    g++
FLAGS = -Wall -g

all: $(EXEC)

$(EXEC): $(SOURCE)
	$(CC) $(FLAGS) -o $(EXEC) $(SOURCE)

valgrind: $(EXEC)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(EXEC) $(ARGS)

run: $(EXEC)
	./$(EXEC) $(ARGS)

clean:
	rm -f $(OBJS) $(EXEC) output.txt
