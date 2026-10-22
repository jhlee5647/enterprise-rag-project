"""부서 매핑: 문서의 IDL 토픽을 연결표와 비교해 허용 부서를 계산한다."""


def allowed_departments(topic: str, mapping: dict[str, str]) -> list[str]:
    """`topic`(예: `Marketing;Youth Prevention;`)에서 연결표에 있는 토픽의 부서를 모은다.

    부서는 이름순으로 정렬하고, 같은 부서는 한 번만 넣는다.
    """
    topics = {t.strip() for t in topic.split(";")}
    return sorted({dept for name, dept in mapping.items() if name in topics})
