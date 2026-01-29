# CROP-Dataset Loader

A Python module for loading and processing the AI4Agr/CROP-dataset from HuggingFace for agricultural QA benchmarking.

## Features

- **HuggingFace Integration**: Downloads and caches the CROP-dataset automatically
- **Smart Filtering**: Filter by crop types, language, or limit the number of samples
- **Structured Output**: Converts raw dataset to structured `QAItem` dataclass format
- **Multi-turn Support**: Handles dialogue history by concatenating into context
- **Crop Detection**: Automatically extracts crop type from question/answer text
- **Language Detection**: Identifies English/Korean text automatically
- **CLI Interface**: Standalone command-line tool for data processing

## Installation

Ensure you have the `datasets` library installed:

```bash
pip install datasets
```

## Usage

### As a Python Module

```python
from benchmarking.data.crop_dataset_loader import load_crop_qa, QAItem

# Load all data
items = load_crop_qa()

# Load with filters
items = load_crop_qa(
    limit=5000,                           # Limit to 5000 items
    crops=['rice', 'corn', 'soybeans'],   # Filter by crop types
    lang='en',                            # Only English questions
    cache_dir='data/crop/raw',            # Custom cache directory
    split='train'                         # Dataset split
)

# Access structured data
for item in items:
    print(f"ID: {item.id}")
    print(f"Question: {item.question}")
    print(f"Answer: {item.answer}")
    print(f"Context: {item.context}")
    print(f"Crop: {item.crop_type}")
    print(f"Language: {item.language}")
```

### As a Command-Line Tool

```bash
# Basic usage
python -m benchmarking.data.crop_dataset_loader --output data/crop --limit 5000

# Filter by crop types
python -m benchmarking.data.crop_dataset_loader \
    --output data/crop \
    --crops rice corn soybeans tomatoes \
    --limit 10000

# Filter by language
python -m benchmarking.data.crop_dataset_loader \
    --output data/crop \
    --lang en \
    --limit 5000

# Show statistics
python -m benchmarking.data.crop_dataset_loader \
    --output data/crop \
    --stats

# Custom cache directory
python -m benchmarking.data.crop_dataset_loader \
    --output data/crop \
    --cache-dir /path/to/cache \
    --limit 5000
```

## Data Schema

### QAItem Dataclass

```python
@dataclass
class QAItem:
    id: str           # Unique identifier (e.g., "crop_0")
    question: str     # Question text (instruction + input)
    answer: str       # Answer text (output)
    context: str      # Dialogue history or system prompt
    crop_type: str    # Detected crop type (e.g., "rice", "corn", "unknown")
    language: str     # Detected language ("en", "ko", "unknown")
```

### Output Format

The module saves data in JSONL format (one JSON object per line):

```json
{"id": "crop_0", "question": "What causes rice blast disease?", "answer": "Rice blast is caused by the fungus Magnaporthe oryzae...", "context": "", "crop_type": "rice", "language": "en"}
{"id": "crop_1", "question": "옥수수 재배 방법은?", "answer": "옥수수는 따뜻한 기후에서...", "context": "[Turn 1]\nQ: 토양 준비는?\nA: 배수가 잘 되는 토양...", "crop_type": "corn", "language": "ko"}
```

## Supported Crop Types

The module automatically detects the following crop types:

- `rice` - Rice/paddy (벼, 쌀)
- `corn` - Corn/maize (옥수수)
- `soybeans` - Soybeans (콩, 대두)
- `tomatoes` - Tomatoes (토마토)
- `wheat` - Wheat (밀)
- `potato` - Potatoes (감자)
- `pepper` - Peppers/chili (고추)
- `cucumber` - Cucumbers (오이)
- `strawberry` - Strawberries (딸기)
- `unknown` - Unidentified crop

## Example Integration

### With BEIR Benchmark Pattern

```python
from benchmarking.data.crop_dataset_loader import load_crop_qa
from core.Models.Schemas import SourceDoc

# Load QA data
qa_items = load_crop_qa(limit=1000, crops=['rice', 'corn'])

# Convert to SourceDoc format for retrieval benchmarking
docs = [
    SourceDoc(
        id=item.id,
        text=item.answer,
        metadata={
            'question': item.question,
            'crop_type': item.crop_type,
            'language': item.language,
        }
    )
    for item in qa_items
]
```

### Save and Load Processed Data

```python
from benchmarking.data.crop_dataset_loader import save_qa_items, load_qa_items
from pathlib import Path

# Save processed data
save_qa_items(items, Path('data/crop/crop_qa.jsonl'))

# Load processed data later
loaded_items = load_qa_items(Path('data/crop/crop_qa.jsonl'))
```

## Dataset Source

- **Name**: AI4Agr/CROP-dataset
- **Platform**: HuggingFace Datasets
- **URL**: https://huggingface.co/datasets/AI4Agr/CROP-dataset
- **License**: Check dataset page for license information

## Raw Dataset Structure

The CROP-dataset contains the following fields:

```python
{
    "instruction": str,  # Main question/instruction
    "input": str,        # Additional input context
    "output": str,       # Answer/response
    "system": str,       # System prompt
    "history": List[List[str]]  # [[q1, a1], [q2, a2], ...]
}
```

## Functions

### Main Functions

- `load_crop_qa()` - Load and filter CROP-dataset from HuggingFace
- `save_qa_items()` - Save QA items to JSONL file
- `load_qa_items()` - Load QA items from JSONL file
- `print_statistics()` - Print dataset statistics

### Helper Functions

- `_extract_crop_type()` - Extract crop type from text
- `_detect_language()` - Detect language from text
- `_concatenate_history()` - Format dialogue history into context

## Error Handling

The module includes comprehensive error handling:

```python
try:
    items = load_crop_qa(limit=5000)
except ImportError:
    print("Please install datasets: pip install datasets")
except Exception as e:
    print(f"Failed to load dataset: {e}")
```

## Logging

The module uses Python's logging module for progress tracking:

```
2026-01-28 14:30:00 - crop_dataset_loader - INFO - Loading CROP-dataset from HuggingFace (split=train)...
2026-01-28 14:30:05 - crop_dataset_loader - INFO - Loaded 15000 raw examples
2026-01-28 14:30:10 - crop_dataset_loader - INFO - Filtered to 5000 QA items (crops=['rice', 'corn'], lang=None, limit=5000)
2026-01-28 14:30:10 - crop_dataset_loader - INFO - Saved 5000 items to data/crop/crop_qa.jsonl
```

## Performance Considerations

- **Caching**: Dataset is cached locally after first download
- **Memory**: Full dataset may be large; use `limit` parameter for testing
- **Filtering**: Filters are applied during iteration to minimize memory usage

## Future Enhancements

Potential improvements for future versions:

- Support for more crop types
- Better language detection (using libraries like `langdetect`)
- Query-document pair generation for retrieval benchmarking
- Integration with existing benchmark pipelines
- Support for validation/test splits

## See Also

- `beir_benchmark.py` - BEIR benchmark runner
- `benchmarking/data/AGENTS.md` - Data module documentation
- `core/Models/Schemas/BaseSchemas.py` - SourceDoc schema
