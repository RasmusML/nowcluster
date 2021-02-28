CC      := gcc
CFLAGS  := -c -Isrc -Wall -Werror -fpic
LDFLAGS := -shared

TARGET  := build/k_means.so

SOURCES := $(wildcard src/*.c)
OBJECTS := $(SOURCES:src/%.c=%.o)

lib-build: lib-compile lib-clean

lib-compile $(TARGET) $(OBJECTS):$(SOURCES) 
	$(CC) $(SOURCES) $(CFLAGS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

lib-clean:
	rm -f $(OBJECTS)

	
