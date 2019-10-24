CC = gcc
IN = main.c src/nn.c
OUT = main.out
CFLAGS = -Wall
LFLAGS = -lm
IFLAGS = -I. -I./include

.SILENT all: clean build run

clean:
	rm -f $(OUT)

build: $(IN) include/main.h include/nn.h
	$(CC) $(IN) -o $(OUT) $(CFLAGS) $(LFLAGS) $(IFLAGS)

run: $(OUT)
	./$(OUT)
