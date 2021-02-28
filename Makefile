CC = gcc
IN = main.c src/nn.c
OUT = main.out
CFLAGS = -Wall
LFLAGS = -lm -lpthread
IFLAGS = -I. -I./include

export ARG_TRAIN=out/questions
export ARG_TEST=/dev/null
export ARG_STOP=data/nltk_stop_words.txt

.SILENT all: clean build run

clean:
	rm -f $(OUT)

build: $(IN)
	$(CC) -g -rdynamic $(IN) -o $(OUT) $(CFLAGS) $(LFLAGS) $(IFLAGS)

run: $(OUT)
	./$(OUT) $(ARG_TRAIN) $(ARG_TEST) $(ARG_STOP) $(ARG_OUT)
