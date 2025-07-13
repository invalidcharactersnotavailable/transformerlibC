CC=gcc
CFLAGS=-Iinclude -Wall -Wextra -Werror -g -O3 -fopenmp -std=c99
LDFLAGS=-lm -fopenmp

# to use CBLAS, run: make USE_CBLAS=1
ifeq ($(USE_CBLAS), 1)
    CFLAGS += -DUSE_CBLAS
    LDFLAGS += -lcblas
endif

# Add json-c support if available
ifeq ($(USE_JSON), 1)
    CFLAGS += -DUSE_JSON_C
    LDFLAGS += -ljson-c
endif

SRC_DIR=src
OBJ_DIR=obj

# source files from src directory
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

release: CFLAGS=-Iinclude -Wall -Wextra -Werror -O3 -DNDEBUG -fopenmp -std=c99
release: LDFLAGS=-lm -fopenmp
release: $(TARGET)

debug: CFLAGS=-Iinclude -Wall -Wextra -g -O0 -fopenmp -std=c99 -DDEBUG
debug: LDFLAGS=-lm -fopenmp
debug: $(TARGET)

# Test targets
test: $(TARGET)
	@echo "Running basic functionality test..."
	./$(TARGET) --test --verbose

test-memory: $(TARGET)
	@echo "Running memory leak test..."
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TARGET) --test

# Documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs
	@echo "# TransformerLibC Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "## API Reference" >> docs/README.md
	@echo "" >> docs/README.md
	@for header in include/*.h; do \
		echo "### $$(basename $$header .h)" >> docs/README.md; \
		echo "" >> docs/README.md; \
		grep -E "^[a-zA-Z_][a-zA-Z0-9_]*\s*\(" $$header | sed 's/^/- /' >> docs/README.md; \
		echo "" >> docs/README.md; \
	done

# Installation
install: $(TARGET)
	@echo "Installing transformer_test..."
	sudo cp $(TARGET) /usr/local/bin/
	sudo chmod +x /usr/local/bin/$(TARGET)

uninstall:
	@echo "Uninstalling transformer_test..."
	sudo rm -f /usr/local/bin/$(TARGET)

# Clean targets
clean:
	rm -f $(TARGET) $(OBJ_DIR)/*.o
	rmdir $(OBJ_DIR) || true

distclean: clean
	rm -rf docs/
	rm -f *.ckpt *.vocab

# Development helpers
format:
	@echo "Formatting code..."
	@find . -name "*.c" -o -name "*.h" | xargs clang-format -i

check:
	@echo "Running static analysis..."
	cppcheck --enable=all --std=c99 include/ src/ main.c

# Examples
example-train:
	@echo "Creating example training data..."
	@mkdir -p examples/data
	@echo "This is an example text file for training." > examples/data/sample.txt
	@echo "It contains multiple lines of text." >> examples/data/sample.txt
	@echo "Each line will be used as a training sample." >> examples/data/sample.txt
	@echo "Example training command:" > examples/train_example.sh
	@echo "./transformer_test --data examples/data/sample.txt --save model.ckpt --epochs 5" >> examples/train_example.sh
	@chmod +x examples/train_example.sh

example-finetune:
	@echo "Creating example finetuning data..."
	@mkdir -p examples/finetune
	@echo '{"input": "Translate to French:", "output": "Traduire en français:"}' > examples/finetune/sample1.json
	@echo '{"input": "What is AI?", "output": "What is artificial intelligence?"}' > examples/finetune/sample2.json
	@echo "Example finetuning command:" > examples/finetune_example.sh
	@echo "./transformer_test --load model.ckpt --finetune_dir examples/finetune/ --save finetuned.ckpt" >> examples/finetune_example.sh
	@chmod +x examples/finetune_example.sh

example-inference:
	@echo "Creating example inference script..."
	@echo "Example inference command:" > examples/inference_example.sh
	@echo "./transformer_test --load model.ckpt --inference --prompt \"Hello world\" --max_length 50" >> examples/inference_example.sh
	@chmod +x examples/inference_example.sh

examples: example-train example-finetune example-inference
	@echo "Examples created in examples/ directory"

# Performance benchmarks
benchmark: $(TARGET)
	@echo "Running performance benchmarks..."
	@echo "Small model (2 layers, 512 dim):"
	./$(TARGET) --test --n_layers 2 --embed_dim 512 --n_heads 8
	@echo ""
	@echo "Medium model (6 layers, 768 dim):"
	./$(TARGET) --test --n_layers 6 --embed_dim 768 --n_heads 12
	@echo ""
	@echo "Large model (12 layers, 1024 dim):"
	./$(TARGET) --test --n_layers 12 --embed_dim 1024 --n_heads 16

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the transformer_test executable"
	@echo "  release      - Build optimized release version"
	@echo "  debug        - Build debug version with symbols"
	@echo "  test         - Run basic functionality tests"
	@echo "  test-memory  - Run memory leak tests with valgrind"
	@echo "  docs         - Generate documentation"
	@echo "  install      - Install to /usr/local/bin"
	@echo "  uninstall    - Remove from /usr/local/bin"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  format       - Format code with clang-format"
	@echo "  check        - Run static analysis"
	@echo "  examples     - Create example scripts and data"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  USE_CBLAS=1  - Enable BLAS acceleration"
	@echo "  USE_JSON=1   - Enable JSON-C support"

.PHONY: all release debug test test-memory docs install uninstall clean distclean format check examples example-train example-finetune example-inference benchmark help 