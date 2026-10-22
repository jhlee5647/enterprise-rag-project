"""#5의 질의응답 평가셋 fixture. 문서 A는 inca0001, B는 pmtb0001, C는 agec0001, D는 crep0001이다."""

Q1 = "influencer campaign budget"
Q2 = "PMTA submission timeline"
Q3 = "influencer campaign budget and creator program results"
Q4 = "age verification policy"


def row(question, doc_ids, department, question_type, keep):
    return {
        "question": question,
        "answer": "",
        "doc_ids": doc_ids,
        "department": department,
        "question_type": question_type,
        "keep": keep,
    }


EVAL_SET = [
    row(Q1, "inca0001", "marketing", "단일 홉", "O"),
    row(Q2, "pmtb0001", "regulatory", "단일 홉", "O"),
    row(Q3, "inca0001;crep0001", "marketing", "다중 홉 구체형", "O"),
    row(Q4, "agec0001", "marketing", "단일 홉", "X"),
]
