OBJS = search.o parameters.o
HEADERS = params.hpp ivfflat.hpp
SOURCE = search.cpp parameters.cpp IVFFLAT.cpp
EXEC = search

ARGS = -ivfflat -type mnist -seed 9 -d input.dat -q query.dat -kclusters 64 -range true -N 5 -o output.txt -nprobe 4 -sample_pq true -R 500

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
