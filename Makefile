CC=gcc
CFLAGS=-Iinclude -Wall -g
LDFLAGS=-lm

SRC_DIR=src
OBJ_DIR=obj

SRCS=$(wildcard $(SRC_DIR)/*.c)
OBJS=$(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

TARGET=transformer_test

all: $(TARGET)

$(TARGET): main.o $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) main.o $(OBJS) $(LDFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main.o

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) main.o $(OBJ_DIR)/*.o
	rmdir $(OBJ_DIR) || true

.PHONY: all clean 