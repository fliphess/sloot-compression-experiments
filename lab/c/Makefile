CC ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic

steno: steno.c
	$(CC) $(CFLAGS) -o $@ steno.c

clean:
	rm -f steno

.PHONY: clean
