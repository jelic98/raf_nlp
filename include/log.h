#ifdef FLAG_LOG
#ifndef H_LOG_INCLUDE
#define H_LOG_INCLUDE

#include "lib.h"

#ifdef FLAG_LOG_FILE
FILE* flog;
#endif

dt_int time_get(struct timespec);
void timestamp();
void echo_color(eColor, dt_int, const dt_char*, ...);
void color_set(eColor);
void memcheck_log(void*, const dt_char*, const dt_char*, dt_int);

#ifdef H_LOG_IMPLEMENT
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
	fprintf(flog, "[%s.%03d] ", s, (dt_int) tv.tv_usec / 1000);
}

void echo_color(eColor color, dt_int replace, const dt_char* format, ...) {
	va_list args;
	va_start(args, format);
#ifdef FLAG_COLOR_LOG
	color_set(GRAY);
#endif
	if(!replace) {
		timestamp();
	}
#ifdef FLAG_COLOR_LOG
	color_set(color);
#endif
	dt_char* f = (dt_char*) calloc(strlen(format) + 1, sizeof(dt_char));
	if(replace) {
		strcat(f, "\r");
		strcat(f, format);
	} else {
		strcpy(f, format);
		strcat(f, "\n");
	}
	vfprintf(flog, f, args);
#ifdef FLAG_COLOR_LOG
	color_set(NONE);
#endif
	va_end(args);
}

#ifdef FLAG_COLOR_LOG
void color_set(eColor color) {
#ifdef FLAG_DEBUG
	color == NONE ? fprintf(flog, "\033[0m") : fprintf(flog, "\033[1;3%dm", color);
#endif
}
#endif

void memcheck_log(void* ptr, const dt_char* file, const dt_char* func, dt_int line) {
	if(!ptr) {
#ifdef FLAG_LOG
		echo_fail(ERROR_MEMORY " @ %s:%s:%d", file, func, line);
#endif
		exit(1);
	}
}
#endif
#endif
#endif
