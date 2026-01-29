@echo off
REM ==============================================================================
REM ERA-SmartFarm-RAG Benchmark Runner (Windows)
REM ==============================================================================
REM
REM One-stop script for running all benchmarking experiments.
REM
REM Usage:
REM   benchmarking\scripts\benchmark.bat [OPTIONS]
REM
REM Options:
REM   --only EXPERIMENTS   Comma-separated: baseline,ablation,domain,edge
REM   --config PATH        Custom config file path
REM   --output PATH        Output directory
REM   --skip-setup         Skip data/model preparation
REM   --verbose            Enable verbose output
REM   --dry-run            Validate without running
REM   --help               Show this help message
REM
REM Examples:
REM   benchmarking\scripts\benchmark.bat
REM   benchmarking\scripts\benchmark.bat --only baseline,ablation
REM
REM ==============================================================================

setlocal enabledelayedexpansion

REM ------------------------------------------------------------------------------
REM Configuration
REM ------------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
set "BENCHMARK_DIR=%SCRIPT_DIR%.."
set "PROJECT_ROOT=%BENCHMARK_DIR%\.."

set "CONFIG_FILE=%BENCHMARK_DIR%\config\benchmark_config.yaml"
set "OUTPUT_DIR="
set "ONLY_EXPERIMENTS="
set "SKIP_SETUP=0"
set "VERBOSE=0"
set "DRY_RUN=0"

REM ------------------------------------------------------------------------------
REM Argument Parsing
REM ------------------------------------------------------------------------------

:parse_args
if "%~1"=="" goto :after_args
if /i "%~1"=="--only" (
    set "ONLY_EXPERIMENTS=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--config" (
    set "CONFIG_FILE=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output" (
    set "OUTPUT_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--skip-setup" (
    set "SKIP_SETUP=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--verbose" (
    set "VERBOSE=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--help" goto :print_help
if /i "%~1"=="-h" goto :print_help

echo Unknown option: %~1
goto :print_help

:after_args

REM ------------------------------------------------------------------------------
REM Main Execution
REM ------------------------------------------------------------------------------

call :print_banner
call :check_environment
if errorlevel 1 exit /b 1

if "%SKIP_SETUP%"=="0" (
    call :setup_data
    if errorlevel 1 exit /b 1
    call :setup_models
    if errorlevel 1 exit /b 1
)

call :run_experiments
exit /b %errorlevel%

REM ------------------------------------------------------------------------------
REM Functions
REM ------------------------------------------------------------------------------

:print_banner
echo.
echo ================================================================
echo   ERA-SmartFarm-RAG Benchmark Suite
echo ================================================================
echo.
goto :eof

:print_help
echo.
echo Usage: benchmark.bat [OPTIONS]
echo.
echo Options:
echo   --only EXPERIMENTS   Comma-separated: baseline,ablation,domain,edge
echo   --config PATH        Custom config file path
echo   --output PATH        Output directory
echo   --skip-setup         Skip data/model preparation
echo   --verbose            Enable verbose output
echo   --dry-run            Validate without running
echo   --help               Show this help message
echo.
echo Examples:
echo   benchmark.bat
echo   benchmark.bat --only baseline,ablation
echo.
exit /b 0

:check_environment
echo [1/5] Environment Check

REM Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo   x Python not found
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VERSION=%%v"
echo   + Python %PY_VERSION%

REM Check required packages
python -c "import numpy; import yaml" >nul 2>&1
if errorlevel 1 (
    echo   ! Installing dependencies...
    python -m pip install -q -r "%PROJECT_ROOT%\requirements.txt"
    if errorlevel 1 (
        echo   x Failed to install dependencies
        exit /b 1
    )
    echo   + Dependencies installed
) else (
    echo   + Required packages installed
)

echo.
goto :eof

:setup_data
echo [2/5] Data Preparation

set "CORPUS_PATH=%PROJECT_ROOT%\..\smartfarm-ingest\output\wasabi_en_ko_parallel.jsonl"
if exist "%CORPUS_PATH%" (
    for /f %%l in ('find /c /v "" ^< "%CORPUS_PATH%"') do set "CORPUS_LINES=%%l"
    echo   + Corpus: !CORPUS_LINES! documents
) else (
    echo   x Corpus not found: %CORPUS_PATH%
    echo     Please run the dataset pipeline first or download the data.
    exit /b 1
)

set "QA_PATH=%PROJECT_ROOT%\..\smartfarm-ingest\output\wasabi_qa_dataset.jsonl"
if exist "%QA_PATH%" (
    for /f %%l in ('find /c /v "" ^< "%QA_PATH%"') do set "QA_LINES=%%l"
    echo   + QA Dataset: !QA_LINES! questions
) else (
    echo   x QA dataset not found: %QA_PATH%
    echo     Please run the dataset pipeline first or download the data.
    exit /b 1
)

echo.
goto :eof

:setup_models
echo [3/5] Model Preparation

python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); print('loaded')" >nul 2>&1
if errorlevel 1 (
    echo   ! Downloading embedding model...
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
    if errorlevel 1 (
        echo   x Model download failed
        exit /b 1
    )
)
echo   + Embedding model ready

echo.
goto :eof

:run_experiments
echo [4/5] Running Experiments

REM Build command
set "CMD=python -m benchmarking.run_benchmark"
set "CMD=%CMD% --config %CONFIG_FILE%"

if not "%OUTPUT_DIR%"=="" (
    set "CMD=%CMD% --output %OUTPUT_DIR%"
)

if not "%ONLY_EXPERIMENTS%"=="" (
    set "CMD=%CMD% --only %ONLY_EXPERIMENTS%"
)

if "%VERBOSE%"=="1" (
    set "CMD=%CMD% --verbose"
)

if "%DRY_RUN%"=="1" (
    set "CMD=%CMD% --dry-run"
)

REM Change to project root and run
cd /d "%PROJECT_ROOT%"

if "%VERBOSE%"=="1" (
    echo   Command: %CMD%
)

REM Execute benchmark
%CMD%
goto :eof
