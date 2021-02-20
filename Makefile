CC = gcc
IN = main.c src/nn.c
OUT = main.out
CFLAGS = -Wall
LFLAGS = -lm -lpthread
IFLAGS = -I. -I./include

export PATH=out/questions

.SILENT all: clean build run

clean:
	rm -f $(OUT)

build: $(IN)
	$(CC) -g -rdynamic $(IN) -o $(OUT) $(CFLAGS) $(LFLAGS) $(IFLAGS)

run: $(OUT)
	./$(OUT) $(PATH)
