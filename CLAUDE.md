## 작업 규칙

- 문서(스펙, 티켓, ADR, `GLOSSARY.md`, PR 본문 등)는 기본적으로 **한국어**로 작성한다. 명령어, 코드, 파일 경로, 라벨 문자열은 원문 그대로 둔다.
- 계획 단계에서 만든 문서 파일은 계획이 확정되면 `main`에 한 번에 push한다. 그 전에는 커밋하거나 push하지 않는다.

## Agent skills

### Issue tracker

스펙과 티켓은 GitHub Issues(`jhlee5647/enterprise-rag-project`, `gh` CLI)에서 관리한다. `docs/agents/issue-tracker.md` 참고.

### Issue and PR format

스펙 이슈, 구현 티켓, PR 본문(`/to-spec`, `/to-tickets`, `/pr`)은 스킬 템플릿 대신 `docs/agents/issue-format.md`의 형식으로 쓴다.

### Triage labels

기본 라벨을 사용한다: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. `docs/agents/triage-labels.md` 참고.

### Domain docs

Single-context 구조: 저장소 루트에 `GLOSSARY.md` 하나와 `docs/adr/`. `docs/agents/domain.md` 참고.
