# =============================================================================
# llm.c Makefile
#
# Targets:
#   make                — build main training binary
#   make tools          — build build_vocab and prepare_data
#   make all            — build everything
#   make test           — build and run all unit tests
#   make debug          — debug build with sanitizers
#   make clean          — remove build artifacts
# =============================================================================

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter
LDFLAGS = -lm

RELEASE_FLAGS = -O3 -march=native -ffast-math -fopenmp
DEBUG_FLAGS   = -O0 -g3 -fsanitize=address,undefined -fno-omit-frame-pointer -fopenmp

SRC_DIR   = src
INC_DIR   = include
TOOL_DIR  = tools
TEST_DIR  = tests
BUILD_DIR = build

LIB_SRCS = $(SRC_DIR)/matrix.c     \
           $(SRC_DIR)/attention.c   \
           $(SRC_DIR)/transformer.c \
           $(SRC_DIR)/tokenizer.c   \
           $(SRC_DIR)/inference.c

TRAIN_SRC = $(SRC_DIR)/train.c \
            $(SRC_DIR)/main.c

# =============================================================================
# Default: training binary
# =============================================================================
.PHONY: train
train: $(BUILD_DIR)/train

$(BUILD_DIR)/train: $(LIB_SRCS) $(TRAIN_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

# =============================================================================
# Tools
# =============================================================================
.PHONY: tools
tools: $(BUILD_DIR)/build_vocab $(BUILD_DIR)/prepare_data $(BUILD_DIR)/generate

$(BUILD_DIR)/build_vocab: $(TOOL_DIR)/build_vocab.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

$(BUILD_DIR)/prepare_data: $(TOOL_DIR)/prepare_data.c $(SRC_DIR)/tokenizer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

$(BUILD_DIR)/generate: $(SRC_DIR)/generate.c $(LIB_SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

# =============================================================================
# Build everything
# =============================================================================
.PHONY: all
all: train tools

# =============================================================================
# Debug builds
# =============================================================================
.PHONY: debug
debug: $(BUILD_DIR)/train_debug $(BUILD_DIR)/prepare_data_debug

$(BUILD_DIR)/train_debug: $(LIB_SRCS) $(TRAIN_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built (debug): $@"

$(BUILD_DIR)/prepare_data_debug: \
		$(TOOL_DIR)/prepare_data.c $(SRC_DIR)/tokenizer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built (debug): $@"

# =============================================================================
# Tests
# =============================================================================
.PHONY: test
test: test_matrix test_tokenizer
	@echo ""
	@echo "=== All tests complete ==="

.PHONY: test_matrix
test_matrix: $(BUILD_DIR)/test_matrix
	@echo ""
	@echo "Running matrix tests..."
	@$(BUILD_DIR)/test_matrix

$(BUILD_DIR)/test_matrix: $(TEST_DIR)/test_matrix.c $(SRC_DIR)/matrix.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: test_tokenizer
test_tokenizer: $(BUILD_DIR)/test_tokenizer
	@echo ""
	@echo "Running tokenizer tests..."
	@$(BUILD_DIR)/test_tokenizer

$(BUILD_DIR)/test_tokenizer: $(TEST_DIR)/test_tokenizer.c $(SRC_DIR)/tokenizer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: test_debug
test_debug: $(BUILD_DIR)/test_matrix_debug $(BUILD_DIR)/test_tokenizer_debug
	@$(BUILD_DIR)/test_matrix_debug
	@$(BUILD_DIR)/test_tokenizer_debug

$(BUILD_DIR)/test_matrix_debug: $(TEST_DIR)/test_matrix.c $(SRC_DIR)/matrix.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/test_tokenizer_debug: $(TEST_DIR)/test_tokenizer.c $(SRC_DIR)/tokenizer.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) $^ -o $@ $(LDFLAGS)

# =============================================================================
# Utilities
# =============================================================================
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned."

.PHONY: loc
loc:
	@echo "Lines of code:"
	@wc -l $(SRC_DIR)/*.c $(INC_DIR)/*.h $(TOOL_DIR)/*.c $(TEST_DIR)/*.c \
	    2>/dev/null | sort -rn | head -20

# Full pipeline shortcut: make sample_run CORPUS=data/corpus.txt
.PHONY: sample_run
sample_run: tools train
	@echo "--- Step 1: Build vocabulary ---"
	mkdir -p data
	./$(BUILD_DIR)/build_vocab $(CORPUS) 4096 data/tokenizer.vocab
	@echo "--- Step 2: Prepare data ---"
	./$(BUILD_DIR)/prepare_data $(CORPUS) data/tokenizer.vocab \
	    --out_dir data --train_split 0.9
	@echo "--- Step 3: Train ---"
	./$(BUILD_DIR)/train

# Benchmark (not part of `make test` — run explicitly)
.PHONY: bench
bench: $(BUILD_DIR)/bench_matmul $(BUILD_DIR)/bench_inference
	@echo ""
	@./$(BUILD_DIR)/bench_matmul
	@echo ""
	@./$(BUILD_DIR)/bench_inference --random

$(BUILD_DIR)/bench_matmul: $(TEST_DIR)/bench_matmul.c $(SRC_DIR)/matrix.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

$(BUILD_DIR)/bench_inference: $(TEST_DIR)/bench_inference.c $(LIB_SRCS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(RELEASE_FLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built: $@"

	@echo "Built: $@"
