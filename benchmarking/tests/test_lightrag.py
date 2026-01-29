"""LightRAG 모듈 테스트.

Docker에서 실행:
    docker-compose run --rm --entrypoint python benchmark-cpu benchmarking/tests/test_lightrag.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarking.bootstrap import ensure_search_on_path

ensure_search_on_path()


def test_import():
    """Import 테스트."""
    print("=" * 60)
    print("1. Import 테스트")
    print("=" * 60)

    try:
        from core.Services.Retrieval.LightRAG import (
            LightRAGRetriever,
            LightRAGRetrieverWithDense,
            create_lightrag_retriever,
        )
        print("[OK] LightRAG 모듈 import 성공")
        print(f"  - LightRAGRetriever: {LightRAGRetriever}")
        print(f"  - LightRAGRetrieverWithDense: {LightRAGRetrieverWithDense}")

        from benchmarking.baselines import LightRAGBaseline
        print("[OK] LightRAGBaseline import 성공")

        return True
    except ImportError as e:
        print(f"[FAIL] Import 실패: {e}")
        return False


def test_lightrag_library():
    """lightrag-hku 라이브러리 설치 확인."""
    print("\n" + "=" * 60)
    print("2. lightrag-hku 라이브러리 테스트")
    print("=" * 60)

    try:
        import lightrag
        print(f"[OK] lightrag 모듈 import 성공")
        print(f"  - 버전: {getattr(lightrag, '__version__', 'unknown')}")
        print(f"  - 경로: {lightrag.__file__}")

        from lightrag import LightRAG, QueryParam
        print("[OK] LightRAG, QueryParam import 성공")

        from lightrag.utils import EmbeddingFunc
        print("[OK] EmbeddingFunc import 성공")

        return True
    except ImportError as e:
        print(f"[WARN] lightrag-hku 미설치: {e}")
        print("  pip install lightrag-hku 로 설치하세요.")
        return False


def test_retriever_creation():
    """Retriever 생성 테스트 (라이브러리 없이)."""
    print("\n" + "=" * 60)
    print("3. LightRAGRetriever 생성 테스트")
    print("=" * 60)

    from core.Services.Retrieval.LightRAG import LightRAGRetriever
    from core.Models.Schemas import SourceDoc

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            retriever = LightRAGRetriever(
                working_dir=tmpdir,
                query_mode="hybrid"
            )
            print(f"[OK] Retriever 생성 성공: {retriever}")

            # 문서 추가 (LLM 없이)
            docs = [
                SourceDoc(id="doc1", text="와사비는 수온 관리가 중요합니다.", metadata={}),
                SourceDoc(id="doc2", text="연부병은 고수온에서 발생합니다.", metadata={}),
                SourceDoc(id="doc3", text="수온은 15-18도가 적정합니다.", metadata={}),
            ]
            retriever._docs = docs
            print(f"[OK] 문서 {len(docs)}개 추가")

            # search 테스트 (키워드 매칭)
            results = retriever.search("연부병 원인", k=2)
            print(f"[OK] search() 결과: {len(results)}개 문서")
            for r in results:
                print(f"  - {r.id}: {r.text[:50]}...")

            return True
        except Exception as e:
            print(f"[FAIL] Retriever 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_baseline_build():
    """LightRAGBaseline 구축 테스트."""
    print("\n" + "=" * 60)
    print("4. LightRAGBaseline 구축 테스트")
    print("=" * 60)

    from benchmarking.baselines import LightRAGBaseline
    from core.Models.Schemas import SourceDoc

    with tempfile.TemporaryDirectory() as tmpdir:
        docs = [
            SourceDoc(id="doc1", text="와사비 재배에서 수온 관리가 중요합니다.", metadata={}),
            SourceDoc(id="doc2", text="연부병은 고수온에서 자주 발생합니다.", metadata={}),
            SourceDoc(id="doc3", text="적정 수온은 15-18도입니다.", metadata={}),
        ]

        try:
            # LightRAG 라이브러리가 있는 경우에만 full 테스트
            import lightrag
            has_lightrag = True
        except ImportError:
            has_lightrag = False

        if has_lightrag:
            try:
                baseline = LightRAGBaseline.build_from_docs(
                    docs=docs,
                    working_dir=tmpdir,
                    query_mode="naive"  # 가장 단순한 모드
                )
                print(f"[OK] LightRAGBaseline 구축 성공: {baseline}")

                results = baseline.search("수온 관리", k=2)
                print(f"[OK] search() 결과: {len(results)}개 문서")

                return True
            except Exception as e:
                print(f"[WARN] LightRAGBaseline 구축 실패 (LLM 서버 필요할 수 있음): {e}")
                return False
        else:
            print("[SKIP] lightrag-hku 미설치로 baseline 구축 테스트 건너뜀")
            return True


def test_with_dense_retriever():
    """Dense Retriever 통합 테스트."""
    print("\n" + "=" * 60)
    print("5. Dense Retriever 통합 테스트")
    print("=" * 60)

    from core.Services.Retrieval.LightRAG import LightRAGRetrieverWithDense
    from core.Services.Retrieval.Embeddings import EmbeddingRetriever
    from core.Models.Schemas import SourceDoc

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Dense retriever 생성
            dense = EmbeddingRetriever(device="cpu", model_id="minilm")

            docs = [
                SourceDoc(id="doc1", text="와사비 재배에서 수온 관리가 중요합니다.", metadata={}),
                SourceDoc(id="doc2", text="연부병은 고수온에서 자주 발생합니다.", metadata={}),
                SourceDoc(id="doc3", text="적정 수온은 15-18도입니다.", metadata={}),
            ]
            dense.build(docs)
            print(f"[OK] Dense retriever 구축 성공")

            # LightRAG + Dense 통합
            retriever = LightRAGRetrieverWithDense(
                working_dir=tmpdir,
                dense_retriever=dense,
                query_mode="hybrid"
            )
            retriever._docs = docs
            print(f"[OK] LightRAGRetrieverWithDense 생성 성공")

            results = retriever.search("연부병 원인", k=2)
            print(f"[OK] search() 결과: {len(results)}개 문서")
            for r in results:
                print(f"  - {r.id}: {r.text[:50]}...")

            return True
        except Exception as e:
            print(f"[WARN] Dense 통합 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """메인 테스트 실행."""
    print("\n" + "#" * 60)
    print("# LightRAG 모듈 테스트")
    print("#" * 60)

    results = {}

    # 1. Import 테스트
    results["import"] = test_import()

    # 2. 라이브러리 테스트
    results["library"] = test_lightrag_library()

    # 3. Retriever 생성 테스트
    results["retriever"] = test_retriever_creation()

    # 4. Baseline 구축 테스트
    results["baseline"] = test_baseline_build()

    # 5. Dense 통합 테스트
    results["dense"] = test_with_dense_retriever()

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n총 {total}개 중 {passed}개 통과")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
