# PathRAG Comparison Test

## Overview
This test directly compares the OLD `PathRAGRetriever` vs NEW `PathRAGLtRetriever` implementations without ontology dependency.

## Files
- `test_pathrag_comparison.py` - Main test implementation
- `run_pathrag_comparison.py` - Wrapper script for easy execution

## Test Approach
1. **Creates a small test graph manually** (no corpus needed)
   - Concept nodes: crop (와사비), disease (흰가루병), env (온도)
   - Practice nodes: 4 different farming practices
   - Intermediate nodes: stage (생육기)

2. **Directly specifies seed nodes** (bypasses ontology matching)
   - Test 1: Direct 1-hop path from `crop:wasabi`
   - Test 2: 2-hop path from `env:temperature` via `stage:growth`

3. **Compares results and scores** from both retrievers

## Running the Test

### Option 1: Direct execution
```bash
cd smartfarm-benchmarking
python -m benchmarking.runners.run_pathrag_comparison
```

### Option 2: Via Python inline
```bash
cd smartfarm-benchmarking
python -c "from benchmarking.tests.test_pathrag_comparison import test_direct_search; test_direct_search()"
```

## Test Results

### Test 1: Direct 1-hop from crop:wasabi
- **Old PathRAG**: 2 results (practice:p1, practice:p3)
- **New PathRAG-lt**: 3 results with scores
  - practice:p1 (score=0.5091) - 온도 관리
  - practice:p3 (score=0.5091) - 양액 관리
  - practice:p4 (score=0.0851) - 습도 관리 (via 2-hop path)

### Test 2: 2-hop path from env:temperature
- **Old PathRAG**: 2 results (practice:p1, practice:p4)
- **New PathRAG-lt**: 2 results with scores
  - practice:p1 (score=0.2365) - 온도 관리
  - practice:p4 (score=0.2365) - 습도 관리

## Key Observations

### 1. **Score-based Ranking**
The new PathRAG-lt provides confidence scores for every result, enabling:
- Better ranking of results
- Threshold-based filtering
- Confidence-aware retrieval

### 2. **Multi-hop Discovery**
In Test 1, the new implementation found an additional practice (p4) via a 2-hop path:
```
crop:wasabi -> stage:growth -> practice:p4
```
This shows the improved exploration capability.

### 3. **Score Decay with Depth**
- 1-hop results: score ≈ 0.51
- 2-hop results: score ≈ 0.24 or 0.09
- Clear decay pattern shows path distance influence

### 4. **Equal Scores for Equal Paths**
When multiple practices have the same path structure, they receive equal scores (0.5091 for p1 and p3 in Test 1).

## Advantages of New PathRAG-lt

1. ✅ **Quantitative scoring** - Every result has a confidence score
2. ✅ **Better ranking** - Can sort by relevance
3. ✅ **Flexible filtering** - Can apply score thresholds
4. ✅ **Multi-path aggregation** - Combines evidence from multiple paths
5. ✅ **Configurable exploration** - Threshold and alpha parameters

## Next Steps

This comparison validates that:
- Both implementations work correctly on simple test graphs
- The new implementation provides additional scoring capabilities
- Path discovery behaves as expected for 1-hop and 2-hop scenarios

For full validation, run the BEIR benchmark comparison to see performance on real retrieval tasks.
