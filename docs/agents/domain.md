# 도메인 문서

엔지니어링 스킬이 코드베이스를 탐색할 때 이 저장소의 도메인 문서를 어떻게 활용하는지 정의한다.

## 탐색 전에 읽을 것

- 저장소 루트의 **`GLOSSARY.md`**, 또는
- 루트에 **`GLOSSARY-MAP.md`**가 있다면 그것. 컨텍스트별 `GLOSSARY.md`를 가리키므로, 주제와 관련된 것을 각각 읽는다.
- **`docs/adr/`**: 작업할 영역과 관련된 ADR을 읽는다. multi-context 저장소에서는 컨텍스트별 결정이 담긴 `src/<context>/docs/adr/`도 확인한다.

이 파일들이 없으면 **아무 말 없이 진행한다**. 없다는 사실을 지적하지 말고, 미리 만들자고 제안하지도 않는다. 용어나 결정이 실제로 확정될 때 `/domain-modeling` 스킬(`/grill-with-docs`, `/improve-codebase-architecture`를 통해 호출됨)이 필요한 시점에 만든다.

## 파일 구조

Single-context 저장소 (대부분의 저장소, 이 저장소 포함):

```
/
├── GLOSSARY.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context 저장소 (루트에 `GLOSSARY-MAP.md`가 있음):

```
/
├── GLOSSARY-MAP.md
├── docs/adr/                          ← 시스템 전체 결정
└── src/
    ├── ordering/
    │   ├── GLOSSARY.md
    │   └── docs/adr/                  ← 컨텍스트별 결정
    └── billing/
        ├── GLOSSARY.md
        └── docs/adr/
```

## 용어집의 어휘를 사용한다

산출물(이슈 제목, 리팩터링 제안, 가설, 테스트 이름 등)에서 도메인 개념을 지칭할 때는 `GLOSSARY.md`에 정의된 용어를 쓴다. 용어집이 명시적으로 피하는 동의어로 바꿔 쓰지 않는다.

필요한 개념이 용어집에 없다면 신호로 받아들인다. 프로젝트가 쓰지 않는 말을 지어내고 있거나(재고할 것), 실제로 빈틈이 있는 것이다(`/domain-modeling`용으로 기록할 것).

## ADR 충돌을 드러낸다

산출물이 기존 ADR과 모순되면 조용히 덮어쓰지 말고 명시적으로 밝힌다.

> _ADR-0007(이벤트 소싱 주문)과 모순되지만, 다음 이유로 재검토할 가치가 있음…_
