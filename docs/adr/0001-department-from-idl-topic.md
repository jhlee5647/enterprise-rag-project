# 부서는 IDL 토픽에서 유도한다

JUUL 컬렉션에는 부서 정보가 없다(`filepath`는 보관자의 메일함 경로뿐이다). 그래서 접근 권한의 단위인 부서를 IDL이 붙인 업무 분야 꼬리표(IDL 토픽)에서 유도한다. 연결은 1:1이다: `Marketing` → `marketing`, `Government & Public Affairs` → `regulatory`, `Product Design & Quality` → `product`. 문서의 허용 부서는 연결되는 IDL 토픽들의 부서를 모두 합친 집합이고, 비어 있으면 그 문서는 다루지 않는다.

## Considered Options

- **우리가 정의한 주제를 부서로 쓰기**(제목 키워드로 정한 marketing/rnd/regulatory): 근거가 우리의 검색어뿐이라 "실제 기업 문서"라는 차별점과 맞지 않는다.
- **보관자 → 부서 매핑**: 실제 기업 ACL에 가장 가깝다. 하지만 보관자에게 직함이나 부서 정보가 없어 매핑의 근거가 약하고, 만드는 비용이 크다.
- **여러 IDL 토픽을 한 부서로 묶기**(예: Health & Safety, Flavors → product): "Youth Prevention은 마케팅인가 규제대응인가" 같은 근거 약한 판단이 들어간다. PoC에서는 1:1로 두고, 연결표를 config로 빼서 나중에 바꿀 수 있게 한다.

## Consequences

- IDL 토픽은 JUUL 컬렉션의 약 16%(약 80만 건)에만 붙어 있으므로, 표본은 토픽이 붙은 문서 쪽으로 편향된다.
- 토픽이 여러 개인 문서가 많다(표본 조사에서 약 40%). 그래서 여러 부서가 함께 보는 문서가 흔하다.
- 연결표에 없는 토픽(Youth Prevention, Flavors, Corporate Development 등)만 가진 문서는 표본에서 빠진다.
- 첨부 문서는 대부분 자기 IDL 토픽이 없다. PoC에서는 첨부를 수집하지 않는다. 다음 단계에서는 첨부가 부모 이메일의 허용 부서를 따르게 하고, 첨부 자기 토픽의 허용 부서가 부모와 다르면 권한 불일치로 보류한 뒤 사람이 결정한다.
