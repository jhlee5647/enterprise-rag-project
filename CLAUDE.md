## 작업 규칙

- 문서(스펙, 티켓, ADR, `GLOSSARY.md`, PR 본문 등)는 기본적으로 **한국어**로 작성한다. 명령어, 코드, 파일 경로, 라벨 문자열은 원문 그대로 둔다.
- 계획 단계에서 만든 문서 파일은 계획이 확정되면 `main`에 한 번에 push한다. 그 전에는 커밋하거나 push하지 않는다.

## 구현 흐름 (`/implement`)

티켓 하나 = 브랜치 하나 = PR 하나. PR 머지는 사용자가 한다.

1. **브랜치**: `main`을 최신으로 맞추고(`git switch main && git pull --ff-only`) `<type>/<티켓 번호>-<영문 slug>` 브랜치를 만든다. 예: `feat/2-collect`. 이전 티켓의 브랜치가 남아 있으면 먼저 4단계부터 한다.
2. **개발**: 모든 커밋을 이 브랜치에 한다.
3. **PR**: 브랜치를 원격에 push하고 `/pr`로 PR을 만든다(본문 끝 `Closes #<티켓 번호>`). PR 링크를 전하고 "머지하셨나요?"라고 물은 뒤 멈춘다.
4. **정리**: 사용자가 머지했다고 하면 `gh pr view <번호> --json state`가 `MERGED`인지 확인한다. `MERGED`이면 `main`으로 전환해 pull하고, 로컬 브랜치와 원격 브랜치를 삭제한다. 다른 상태이면 브랜치를 그대로 두고 확인된 상태를 알린다.

## Agent skills

### Issue tracker

스펙과 티켓은 GitHub Issues(`jhlee5647/enterprise-rag-project`, `gh` CLI)에서 관리한다. `docs/agents/issue-tracker.md` 참고.

### Issue and PR format

스펙 이슈, 구현 티켓, PR 본문(`/to-spec`, `/to-tickets`, `/pr`)은 스킬 템플릿 대신 `docs/agents/issue-format.md`의 형식으로 쓴다. 새 기능과 개선은 형식이 다르다. `/code-review`의 Spec 검토도 이 문서에 따라 스펙을 읽는다.

### Triage labels

기본 라벨을 사용한다: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. `docs/agents/triage-labels.md` 참고.

### Domain docs

Single-context 구조: 저장소 루트에 `GLOSSARY.md` 하나와 `docs/adr/`. `docs/agents/domain.md` 참고.
