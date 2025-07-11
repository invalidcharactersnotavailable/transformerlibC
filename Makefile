CC=gcc
CFLAGS=-Iinclude -Wall -g -O3 -fopenmp
LDFLAGS=-lm -fopenmp

# To use CBLAS, run: make USE_CBLAS=1
ifeq ($(USE_CBLAS), 1)
    CFLAGS += -DUSE_CBLAS
    LDFLAGS += -lcblas
endif

SRC_DIR=src
OBJ_DIR=obj

# Source files from src directory
APP_SRC = $(wildcard $(SRC_DIR)/*.c)
# main.c is separate
MAIN_SRC = main.c

# Object files for src files
APP_OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(APP_SRC))
# Object file for main.c
MAIN_OBJ = $(patsubst %.c, $(OBJ_DIR)/%.o, $(MAIN_SRC))

OBJ = $(APP_OBJ) $(MAIN_OBJ)

TARGET=transformer_test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# Rule for compiling sources in src directory
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for compiling main.c
$(OBJ_DIR)/main.o: main.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c main.c -o $(OBJ_DIR)/main.o

clean:
	rm -f $(TARGET) $(OBJ_DIR)/*.o
	rmdir $(OBJ_DIR) || true

.PHONY: all clean 