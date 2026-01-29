<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-01-28 | Updated: 2026-01-28 -->

# data/

평가 데이터셋 저장소.

## Key Files
| File | Description |
|------|-------------|
| `smartfarm_eval.jsonl` | SmartFarm QA 평가 쿼리 데이터셋 |
| `crop_dataset_loader.py` | CROP-dataset loader module for agricultural QA |
| `causal_extraction_gold.jsonl` | Gold standard for causal extraction benchmarks |
| `multihop_gold.jsonl` | Gold standard for multi-hop reasoning benchmarks |

## smartfarm_eval.jsonl

SmartFarm 도메인 QA 평가용 쿼리 데이터셋.

### Format

```json
{
  "id": "q001",
  "question": "토마토 온실 생육 최적 온도는?",
  "category": "온도",
  "complexity": "easy"
}
```

### Categories

| Category | Description |
|----------|-------------|
| 온도 | Temperature management |
| 양액 | Nutrient solution |
| 병해충 | Pests & diseases |
| 재배일정 | Cultivation schedule |
| 수확 | Harvesting |
| 저장 | Storage |

### Complexity Levels

- **easy**: 단순 사실 질문 (1-2개 관련 문서)
- **medium**: 다단계 추론 (3-5개 관련 문서)
- **hard**: 복합 종합 (5개 이상 문서, 추론 필요)

## crop_dataset_loader.py

CROP-dataset loader module for loading agricultural QA data from HuggingFace.

### Quick Start

```python
from benchmarking.data.crop_dataset_loader import load_crop_qa

# Load dataset with filters
items = load_crop_qa(limit=5000, crops=['rice', 'corn'], lang='en')

# Each item has: id, question, answer, context, crop_type, language
for item in items:
    print(f"Q: {item.question}")
    print(f"A: {item.answer}")
```

### CLI Usage

```bash
python -m benchmarking.data.crop_dataset_loader \
    --output data/crop \
    --limit 5000 \
    --crops rice corn soybeans \
    --stats
```

### Documentation

- `README_CROP.md` - Complete usage guide
- `CROP_LOADER_STATUS.md` - Implementation status and known issues
- `example_crop_usage.py` - Usage examples

## For AI Agents

### Loading SmartFarm Data

```python
from benchmarking.utils.experiment_utils import load_queries
from pathlib import Path

queries = load_queries(Path("benchmarking/data/smartfarm_eval.jsonl"))
```

### Loading CROP Dataset

```python
from benchmarking.data.crop_dataset_loader import load_crop_qa

items = load_crop_qa(limit=1000, crops=['rice', 'tomatoes'])
```

### Direct Loading

```python
import json
from pathlib import Path

data_file = Path("benchmarking/data/smartfarm_eval.jsonl")
with data_file.open(encoding="utf-8") as f:
    queries = [json.loads(line) for line in f]
```

## Integration Points

- **실험**: `benchmarking/experiments/` 스크립트에서 로드
- **유틸리티**: `benchmarking/utils/experiment_utils.py`의 `load_queries()` 사용
- **러너**: `benchmarking/runners/` 스크립트에서 `--input` 인자로 전달

## Conventions

- UTF-8 인코딩
- JSONL 형식 (한 줄에 하나의 JSON 객체)
- `id` 필드는 고유해야 함
- `category`와 `complexity` 필드로 그룹별 분석 가능

## Notes

- 기존 `datasets/smartfarm-qa-bench/` 구조 대신 단일 파일로 통합됨
- 새 쿼리 추가 시 JSONL 파일에 한 줄씩 추가
- 카테고리와 복잡도 필드는 도메인 분석(RQ3)에 활용
