CC      := gcc
CFLAGS  := -c -Wall -Werror -fopenmp -fpic -O2
LDFLAGS := -shared -fopenmp -O2

TARGET  := build/nowcluster.so

SOURCES := $(wildcard *.c)
OBJECTS := $(SOURCES:%.c=%.o)

lib-build: lib-compile lib-clean

lib-compile $(TARGET) $(OBJECTS):$(SOURCES) 
	$(CC) $(SOURCES) $(CFLAGS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

lib-clean:
	rm -f $(OBJECTS)

	
