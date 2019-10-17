CC = gcc
IN = main.c
OUT = main.out
CFLAGS = -O
LFLAGS = -lm

.SILENT all: clean build run

clean:
	rm -f $(OUT)

build: $(IN)
	$(CC) $(IN) -o $(OUT) $(CFLAGS) $(LFLAGS)

run: $(OUT)
	./$(OUT)
