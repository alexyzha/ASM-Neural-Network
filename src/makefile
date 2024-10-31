OUT = ./obj/network
OBJ = ./obj
SRC = network.asm
NASM_FLAGS = -f elf64
GCC_FLAGS = -m64 -nostartfiles -no-pie -lc

all: $(OBJ) $(OUT) run

$(OBJ):
	mkdir -p $(OBJ)

$(OUT).o: $(SRC) | $(OBJ)
	nasm $(NASM_FLAGS) -o $(OUT).o $(SRC)

$(OUT): $(OUT).o
	gcc $(GCC_FLAGS) -o $(OUT) $(OUT).o -lm

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT) $(OUT).o
