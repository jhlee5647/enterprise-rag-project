"""`collect` → `index` → `gen-eval`: 부서마다 마스킹된 청크로 평가셋 초안 CSV를 만든다.

RAGAS 호출(`gen_eval.synthesize`)은 RAGAS 출력과 같은 모양의 가짜로 바꾼다.
RAGAS가 만드는 질문은 실행할 때마다 달라서, 여기서는 질문 내용이 아니라 그 앞뒤를 확인한다.
"""

import pytest

from enterprise_rag import gen_eval

SINGLE = "single_hop_specific_query_synthesizer"
MULTI_SPECIFIC = "multi_hop_specific_query_synthesizer"
MULTI_ABSTRACT = "multi_hop_abstract_query_synthesizer"


class FakeRagas:
    """받은 청크로 유형마다 질문 하나씩을 RAGAS `Testset.to_list()` 모양으로 돌려준다.

    다중 홉 질문은 첫 청크와 마지막 청크를 엮는다. RAGAS처럼 근거 앞에 `<n-hop>`을 붙인다.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, config, chunks, size):
        self.calls.append(chunks)
        first, last = chunks[0], chunks[-1]
        hops = [f"<1-hop>\n\n{first}", f"<2-hop>\n\n{last}"]
        return [
            sample("single question", [first], SINGLE),
            sample("multi specific question", hops, MULTI_SPECIFIC),
            sample("multi abstract question", hops, MULTI_ABSTRACT),
        ]


def sample(question, contexts, synthesizer):
    return {
        "user_input": question,
        "reference_contexts": contexts,
        "reference": f"answer to {question}",
        "synthesizer_name": synthesizer,
    }


@pytest.fixture
def ragas(monkeypatch):
    fake = FakeRagas()
    monkeypatch.setattr(gen_eval, "synthesize", fake)
    return fake


def test_writes_eval_set_csv_with_all_three_departments(pipeline, ragas):
    rows = pipeline.gen_eval()

    assert {r["department"] for r in rows} == {"marketing", "regulatory", "product"}
    assert all(r["keep"] == "" for r in rows)


def test_answer_doc_ids_are_in_document_table_and_allowed_for_the_row_department(pipeline, ragas):
    rows = pipeline.gen_eval()
    allowed = {d["doc_id"]: d["departments"] for d in pipeline.documents()}

    for r in rows:
        doc_ids = r["doc_ids"].split(";")
        assert doc_ids != [""]
        for doc_id in doc_ids:
            assert r["department"] in allowed[doc_id]


def test_multi_hop_question_gets_doc_ids_of_every_hop_and_types_are_named(pipeline, ragas):
    rows = pipeline.gen_eval()

    marketing = {r["question_type"]: r for r in rows if r["department"] == "marketing"}
    assert set(marketing) == {"single_hop", "multi_hop_specific", "multi_hop_abstract"}
    # 가짜 RAGAS의 다중 홉 질문은 첫 청크(문서 ID순 첫 문서)와 마지막 청크(마지막 문서)를 엮는다
    first, *_, last = sorted(
        d["doc_id"] for d in pipeline.documents() if "marketing" in d["departments"]
    )
    assert marketing["single_hop"]["doc_ids"] == first
    assert marketing["multi_hop_specific"]["doc_ids"] == f"{first};{last}"


def test_questions_come_from_at_most_docs_per_department_documents(pipeline, ragas):
    rows = pipeline.gen_eval(docs_per_department=1)

    for department in ("marketing", "regulatory", "product"):
        used = {i for r in rows if r["department"] == department for i in r["doc_ids"].split(";")}
        assert len(used) == 1
    # RAGAS에 넘긴 청크도 고른 문서 1건의 청크뿐이다
    for chunks, department in zip(ragas.calls, ("marketing", "regulatory", "product"), strict=True):
        [doc_id] = {r["doc_ids"] for r in rows if r["department"] == department}
        assert chunks == pipeline.chunks(doc_id)


def test_question_whose_context_matches_no_chunk_is_dropped_and_logged(
    pipeline, ragas, monkeypatch, caplog
):
    # 예: RAGAS가 근거 청크를 고쳐 써서 색인의 어느 청크와도 같지 않은 경우
    def with_unmatched(config, chunks, size):
        return ragas(config, chunks, size) + [
            sample("rewritten question", ["text that is in no chunk"], SINGLE)
        ]

    monkeypatch.setattr(gen_eval, "synthesize", with_unmatched)
    rows = pipeline.gen_eval()

    assert "rewritten question" not in {r["question"] for r in rows}
    assert len(rows) == 9
    assert "문서 ID를 되찾지 못한 질문 1개를 뺐다 (marketing)" in caplog.text


def test_question_row_recovers_doc_ids_from_ragas_multi_hop_contexts():
    doc_of = {"chunk of A": "inca0001", "chunk of D": "crep0001"}
    ragas_row = {
        "user_input": "How did the campaign goal relate to the reported results?",
        "reference_contexts": ["<1-hop>\n\nchunk of A", "<2-hop>\n\nchunk of D"],
        "reference": "The goal was reach; the report showed higher reach.",
        "synthesizer_name": "multi_hop_specific_query_synthesizer",
    }

    row = gen_eval.question_row(ragas_row, "marketing", doc_of)

    assert row == {
        "question": "How did the campaign goal relate to the reported results?",
        "answer": "The goal was reach; the report showed higher reach.",
        "doc_ids": "inca0001;crep0001",
        "department": "marketing",
        "question_type": "multi_hop_specific",
        "keep": "",
    }


def test_eval_reads_generated_csv_and_runs_only_rows_marked_keep(pipeline, ragas):
    rows = pipeline.gen_eval()
    rows[0]["keep"], rows[4]["keep"] = "O", "O"

    run = pipeline.eval(rows, "from-gen-eval")

    assert {r["question"] for r in run.records} == {rows[0]["question"], rows[4]["question"]}
    assert len(run.records) == 6
