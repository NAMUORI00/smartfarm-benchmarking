#!/bin/bash
# ==============================================================================
# ERA-SmartFarm-RAG Benchmark Runner (Linux/Mac)
# ==============================================================================
#
# One-stop script for running all benchmarking experiments.
#
# Usage:
#   ./benchmarking/scripts/benchmark.sh [OPTIONS]
#
# Options:
#   --only EXPERIMENTS   Comma-separated: baseline,ablation,domain,edge
#   --config PATH        Custom config file path
#   --output PATH        Output directory
#   --docker             Run in Docker container
#   --skip-setup         Skip data/model preparation
#   --verbose            Enable verbose output
#   --dry-run            Validate without running
#   --help               Show this help message
#
# Examples:
#   # Run all experiments
#   ./benchmarking/scripts/benchmark.sh
#
#   # Run only baseline and ablation
#   ./benchmarking/scripts/benchmark.sh --only baseline,ablation
#
#   # Run with Docker
#   ./benchmarking/scripts/benchmark.sh --docker
#
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$BENCHMARK_DIR/.." && pwd)"

CONFIG_FILE="$BENCHMARK_DIR/config/benchmark_config.yaml"
OUTPUT_DIR=""
ONLY_EXPERIMENTS=""
USE_DOCKER=false
SKIP_SETUP=false
VERBOSE=false
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

print_banner() {
    echo ""
    echo "================================================================"
    echo "  ERA-SmartFarm-RAG Benchmark Suite"
    echo "================================================================"
    echo ""
}

print_help() {
    head -40 "$0" | tail -35 | sed 's/^# //' | sed 's/^#//'
}

print_step() {
    local step=$1
    local total=$2
    local message=$3
    echo -e "${BLUE}[$step/$total]${NC} $message"
}

print_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "  ${RED}✗${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}!${NC} $1"
}

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --only)
                ONLY_EXPERIMENTS="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --skip-setup)
                SKIP_SETUP=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_help
                exit 1
                ;;
        esac
    done
}

# ------------------------------------------------------------------------------
# Environment Check
# ------------------------------------------------------------------------------

check_environment() {
    print_step 1 5 "Environment Check"
    
    # Check Python
    if command -v python3 &> /dev/null; then
        PY_CMD="python3"
    elif command -v python &> /dev/null; then
        PY_CMD="python"
    else
        print_error "Python not found"
        exit 1
    fi
    
    PY_VERSION=$($PY_CMD --version 2>&1 | cut -d' ' -f2)
    print_success "Python $PY_VERSION"
    
    # Check if in virtual environment or has required packages
    if $PY_CMD -c "import numpy; import yaml; import faiss" 2>/dev/null; then
        print_success "Required packages installed"
    else
        print_warning "Some packages may be missing. Installing..."
        if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
            $PY_CMD -m pip install -q -r "$PROJECT_ROOT/requirements.txt"
            print_success "Dependencies installed"
        else
            print_error "requirements.txt not found"
            exit 1
        fi
    fi
    
    # Check disk space (need at least 2GB)
    if command -v df &> /dev/null; then
        AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
        if [[ "$AVAILABLE_GB" -lt 2 ]]; then
            print_warning "Low disk space: ${AVAILABLE_GB}GB available"
        else
            print_success "Disk space: ${AVAILABLE_GB}GB available"
        fi
    fi
    
    echo ""
}

# ------------------------------------------------------------------------------
# Data Setup
# ------------------------------------------------------------------------------

setup_data() {
    if $SKIP_SETUP; then
        return 0
    fi
    
    print_step 2 5 "Data Preparation"
    
    # Check corpus
    CORPUS_PATH="$PROJECT_ROOT/../smartfarm-ingest/output/wasabi_en_ko_parallel.jsonl"
    if [[ -f "$CORPUS_PATH" ]]; then
        CORPUS_LINES=$(wc -l < "$CORPUS_PATH")
        print_success "Corpus: $CORPUS_LINES documents"
    else
        print_error "Corpus not found: $CORPUS_PATH"
        echo "  Please run the dataset pipeline first or download the data."
        exit 1
    fi
    
    # Check QA dataset
    QA_PATH="$PROJECT_ROOT/../smartfarm-ingest/output/wasabi_qa_dataset.jsonl"
    if [[ -f "$QA_PATH" ]]; then
        QA_LINES=$(wc -l < "$QA_PATH")
        print_success "QA Dataset: $QA_LINES questions"
    else
        print_error "QA dataset not found: $QA_PATH"
        echo "  Please run the dataset pipeline first or download the data."
        exit 1
    fi
    
    echo ""
}

# ------------------------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------------------------

setup_models() {
    if $SKIP_SETUP; then
        return 0
    fi
    
    print_step 3 5 "Model Preparation"
    
    # Check/download embedding model
    $PY_CMD -c "
from sentence_transformers import SentenceTransformer
import sys
try:
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print('Embedding model loaded (cached)')
except Exception as e:
    print(f'Downloading embedding model...')
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print('Embedding model downloaded')
" 2>/dev/null && print_success "Embedding model ready" || print_error "Model setup failed"
    
    echo ""
}

# ------------------------------------------------------------------------------
# Run Experiments
# ------------------------------------------------------------------------------

run_experiments() {
    print_step 4 5 "Running Experiments"
    
    # Build command
    CMD="$PY_CMD -m benchmarking.run_benchmark"
    CMD="$CMD --config $CONFIG_FILE"
    
    if [[ -n "$OUTPUT_DIR" ]]; then
        CMD="$CMD --output $OUTPUT_DIR"
    fi
    
    if [[ -n "$ONLY_EXPERIMENTS" ]]; then
        CMD="$CMD --only $ONLY_EXPERIMENTS"
    fi
    
    if $VERBOSE; then
        CMD="$CMD --verbose"
    fi
    
    if $DRY_RUN; then
        CMD="$CMD --dry-run"
    fi
    
    # Change to project root and run
    cd "$PROJECT_ROOT"
    
    if $VERBOSE; then
        echo "  Command: $CMD"
    fi
    
    # Run the benchmark
    $CMD
}

# ------------------------------------------------------------------------------
# Docker Mode
# ------------------------------------------------------------------------------

run_docker() {
    print_banner
    print_step 1 2 "Building Docker Image"
    
    cd "$BENCHMARK_DIR/docker"
    
    if [[ ! -f "Dockerfile" ]]; then
        print_error "Dockerfile not found in $BENCHMARK_DIR/docker/"
        exit 1
    fi
    
    docker build -t era-benchmark:latest -f benchmarking/docker/Dockerfile "$PROJECT_ROOT"
    print_success "Docker image built"
    
    print_step 2 2 "Running Benchmark in Docker"
    
    # Build docker run command
    DOCKER_CMD="docker run --rm"
    DOCKER_CMD="$DOCKER_CMD -v $PROJECT_ROOT/output:/app/output"
    DOCKER_CMD="$DOCKER_CMD -v $PROJECT_ROOT/../smartfarm-ingest/output:/app/data:ro"
    
    if [[ -n "$ONLY_EXPERIMENTS" ]]; then
        DOCKER_CMD="$DOCKER_CMD era-benchmark:latest --only $ONLY_EXPERIMENTS"
    else
        DOCKER_CMD="$DOCKER_CMD era-benchmark:latest"
    fi
    
    echo "  Running: $DOCKER_CMD"
    $DOCKER_CMD
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

main() {
    parse_args "$@"
    
    if $USE_DOCKER; then
        run_docker
        exit 0
    fi
    
    print_banner
    check_environment
    setup_data
    setup_models
    run_experiments
}

main "$@"
