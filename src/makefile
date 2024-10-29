OUT = ./obj/network
OBJ = ./obj
SRC = network.asm
NASM_FLAGS = -f elf64
LD_FLAGS = -m elf_x86_64

all: $(OBJ) $(OUT) run

$(OUT).o: $(SRC) | $(OBJ)
	nasm $(NASM_FLAGS) -o $(OUT).o $(SRC)

$(OUT): $(OUT).o
	ld $(LD_FLAGS) -o $(OUT) $(OUT).o

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT) $(OUT).o
