# 이슈 트래커: GitHub

이 저장소의 이슈와 스펙은 GitHub 이슈로 관리한다. 모든 작업에 `gh` CLI를 사용한다.

- **저장소**: `jhlee5647/enterprise-rag-project`
- 로컬 작업 디렉터리는 아직 이 저장소의 clone이 아니므로, 모든 `gh` 명령에 `-R jhlee5647/enterprise-rag-project`를 붙인다. (clone 후에는 생략 가능)
- 이슈 제목과 본문은 한국어로 작성한다.

## 규칙

- **이슈 생성**: `gh issue create --title "..." --body "..."`. 여러 줄 본문은 heredoc을 사용한다.
- **이슈 조회**: `gh issue view <number> --json number,title,body,labels,comments`.
- **이슈 목록**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`. 필요에 따라 `--label`, `--state` 필터를 붙인다.
- **상위 이슈의 하위 이슈로 연결**: `gh issue create --parent <parent> ...` 또는 생성 후 `gh issue edit <parent> --add-sub-issue <child>` (`gh` 2.94 이상). 구버전 `gh`: `gh api --method POST repos/<owner>/<repo>/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>` (database id, 아래 **차단 관계** 참고). 하위 이슈 기능을 쓸 수 없으면 하위 이슈 본문 맨 위에 `Part of #<parent>`를 적는다.
- **댓글 작성**: `gh issue comment <number> --body "..."`
- **라벨 추가 / 제거**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **닫기**: `gh issue close <number> --comment "..."`

## PR을 트리아지 대상으로 다루기

**PR을 요청 채널로 사용: 아니오.** _(이 저장소가 외부 PR을 기능 요청으로 취급한다면 `yes`로 바꾼다. `/triage`가 이 값을 읽는다.)_

`yes`일 때는 PR도 이슈와 같은 라벨과 상태를 거치며, `gh pr` 명령을 사용한다.

- **PR 조회**: `gh pr view <number> --comments`, diff는 `gh pr diff <number>`.
- **트리아지할 외부 PR 목록**: `gh api --paginate 'repos/{owner}/{repo}/pulls?state=open' --jq '.[] | select(.author_association | IN("OWNER","MEMBER","COLLABORATOR") | not) | {number, title, author: .user.login, author_association, labels: [.labels[].name]}'`.
- **댓글 / 라벨 / 닫기**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub는 이슈와 PR이 같은 번호 체계를 공유하므로 `#42`는 둘 중 어느 것일 수도 있다. `gh pr view 42`로 확인하고, 실패하면 `gh issue view 42`를 사용한다.

## 스킬이 "이슈 트래커에 게시"하라고 할 때

GitHub 이슈를 생성한다.

## 스킬이 "관련 티켓을 가져오라"고 할 때

위 **이슈 조회** 방식으로 읽는다.

## Wayfinding 작업

`/wayfinder`에서 사용한다. **맵(map)**은 하나의 이슈이고, **하위(child)** 이슈가 티켓이다.

- **맵**: `wayfinder:map` 라벨이 붙은 이슈 하나. 본문에 Notes / Decisions-so-far / Fog를 담는다. `gh issue create --label wayfinder:map`.
- **하위 티켓**: GitHub 하위 이슈로 맵에 연결된 이슈 (**상위 이슈의 하위 이슈로 연결** 참고). 하위 이슈 기능이 없으면 맵 본문의 작업 목록에 추가하고 하위 이슈 본문 맨 위에 `Part of #<map>`을 적는다. 라벨: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). 선점(claim)되면 작업 중인 개발자에게 할당한다.
- **차단 관계**: GitHub **기본 이슈 의존성(issue dependencies)**을 표준으로 사용한다(UI에 표시됨). `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`로 추가한다. `<blocker-db-id>`는 차단 이슈의 숫자 **database id**다(`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, `#number`나 `node_id`가 _아님_). GitHub는 `issue_dependencies_summary.blocked_by`(열린 차단 이슈 수, 실시간 기준)를 제공한다. 의존성 기능이 없으면 하위 이슈 본문 맨 위에 `Blocked by: #<n>, #<n>` 줄을 적는다. 모든 차단 이슈가 닫히면 차단이 풀린다.
- **프런티어 조회**: 맵의 열린 하위 이슈를 나열하고(`gh issue list --state open`, 맵의 하위 이슈/작업 목록으로 한정), 열린 차단 이슈가 있거나(`issue_dependencies_summary.blocked_by > 0` 또는 `Blocked by` 줄의 열린 이슈) 담당자가 있는 것은 제외한다. 맵 순서상 첫 번째가 우선이다.
- **선점(claim)**: `gh issue edit <n> --add-assignee @me`. 세션의 첫 번째 쓰기 작업이어야 한다.
- **해결(resolve)**: `gh issue comment <n> --body "<answer>"`, 이어서 `gh issue close <n>`, 그다음 맵의 Decisions-so-far에 맥락 포인터(요지 + 링크)를 추가한다.
