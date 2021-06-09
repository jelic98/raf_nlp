#ifndef H_LOG_INCLUDE
#define H_LOG_INCLUDE

#include "lib.h"

struct timespec time_start;

void sigget(dt_int);
dt_int time_get(struct timespec);
void timestamp();
void echo_color(eColor, dt_int, const dt_char*, ...);
void color_set(eColor);
void memcheck_log(void*, const dt_char*, const dt_char*, dt_int);

#ifdef H_LOG_IMPLEMENT
// Print debug backtrace
void sigget(dt_int sig) {
	void* ptrs[BACKTRACE_DEPTH];
	size_t size = backtrace(ptrs, BACKTRACE_DEPTH);
	dt_char** stack = backtrace_symbols(ptrs, size);
	dt_int i;
	for(i = 0; i < size; i++) {
		stack[i][strchr(stack[i] + 4, ' ') - stack[i]] = '\0';
		echo_fail("%s @ %s", stack[i] + 4, stack[i] + 40);
	}
	free(stack);
	exit(1);
}

dt_int time_get(struct timespec start) {
	struct timespec finish;
	clock_gettime(CLOCK_MONOTONIC, &finish);
	return finish.tv_sec - start.tv_sec;
}

void timestamp() {
	struct timeval tv;
	dt_char s[20];
	gettimeofday(&tv, NULL);
	strftime(s, sizeof(s) / sizeof(*s), "%d-%m-%Y %H:%M:%S", gmtime(&tv.tv_sec));
	printf("[%s.%03d] ", s, (dt_int) tv.tv_usec / 1000);
}

void echo_color(eColor color, dt_int replace, const dt_char* format, ...) {
	va_list args;
	va_start(args, format);
	color_set(GRAY);
	if(!replace) {
		timestamp();
	}
	color_set(color);
	dt_char* f = (dt_char*) calloc(strlen(format) + 1, sizeof(dt_char));
	if(replace) {
		strcat(f, "\r");
		strcat(f, format);
	} else {
		strcpy(f, format);
		strcat(f, "\n");
	}
	vprintf(f, args);
	color_set(NONE);
	va_end(args);
}

void color_set(eColor color) {
#ifdef FLAG_DEBUG
	color == NONE ? printf("\033[0m") : printf("\033[1;3%dm", color);
#endif
}

void memcheck_log(void* ptr, const dt_char* file, const dt_char* func, dt_int line) {
	if(!ptr) {
		echo_fail(ERROR_MEMORY " @ %s:%s:%d", file, func, line);
		exit(1);
	}
}
#endif
#endif
