CC = gcc
IN = main.c src/nn.c
OUT = main.out
CFLAGS = -Wall
LFLAGS = -lm -lpthread
IFLAGS = -I. -I./include

export ARG_TRAIN=/dev/null
export ARG_TEST=/dev/null
export ARG_STOP=/dev/null

.SILENT all: clean build run

clean:
	rm -f $(OUT)

build: $(IN)
	$(CC) -g -rdynamic $(IN) -o $(OUT) $(CFLAGS) $(LFLAGS) $(IFLAGS)

run: $(OUT)
	./$(OUT) $(ARG_TRAIN) $(ARG_TEST) $(ARG_STOP)
