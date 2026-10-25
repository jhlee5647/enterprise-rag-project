# IDL 저장된 응답 fixture

테스트용 가짜 Solr·S3 서버(`tests/fake_idl.py`)가 돌려주는 데이터.

- `solr_docs.json`: Solr 응답의 `response.docs`에 들어가는 문서 레코드. 필드 구성과 값의 모양(`type`은 목록, `topic`은 `;`로 이은 문자열, 인명 필드 등)은 2026-10 실제 IDL Solr 응답을 따랐고, 제목·인명·Bates 번호는 합성 값이다.
- `s3/<문서 ID>.ocr|.txt`: S3 본문 파일. 원문 재배포를 피하려고 합성 텍스트를 쓴다.

| 문서 ID | 문서 종류 | IDL 토픽 | 본문 | 쓰임 |
|---|---|---|---|---|
| mkea0001 | email | `Marketing` | .ocr | 기본 경로, 인명 메타데이터(`Morgan, Kate`) |
| mkea0002 | email | `Marketing;Youth Prevention;` | .ocr | 연결표에 없는 토픽 무시 |
| mkea0003 | email | `Marketing` | .ocr | 할당량 초과 후보 |
| shrm0001 | email | `Marketing; Government & Public Affairs` | .ocr | 합집합, 두 칸 중복 |
| rega0001 | email | `Government & Public Affairs;` | .ocr | regulatory 기본 경로 |
| txtd0001 | document | `Government & Public Affairs` | .txt만 | 본문 대체 경로 |
| prdp0001 | presentation | `product design & quality;` | .ocr | 표기 섞임 |
| flav0001 | document | `Flavors` | .ocr | 허용 부서 빔 |
| notp0001 | document | (필드 없음) | .ocr | 허용 부서 빔 |
| nobd0001 | presentation | `Marketing` | 없음 | 본문 없음 |
| inca0001 | document | `Marketing` | .ocr | 질의응답 문서 A (marketing 전용) |
| pmtb0001 | document | `Government & Public Affairs` | .ocr | 질의응답 문서 B (regulatory 전용) |
| agec0001 | document | `Marketing;Government & Public Affairs` | .ocr | 질의응답 문서 C (두 부서) |
| pnot0001 | presentation | `Marketing` | .ocr | PII 마스킹 문서 P (제목·본문에 실명, 이메일, 전화번호) |
| jdoe0001 | email | `Government & Public Affairs` | .ocr | `@juullabs.com` 이메일 마스킹 |
| fdar0001 | document | `Product Design & Quality` | .ocr | PII 없는 본문은 그대로 (기관명 FDA) |
| mska0001~mska0011 | document | `Product Design & Quality` | .ocr | 마스커 사례 #19 T1~T11 (본문은 사례 문장 하나, 끝 줄바꿈 없음) |
| mskx0001 | document | `Product Design & Quality` | .ocr | 마스커 #19 추가 사례 (라틴 확장 문자 이름) |
| mskb0001~mskb0010 | document | `Product Design & Quality` | .ocr | 마스커 사례 #38 T1~T9와 추가 사례 (본문은 사례 문장 하나, 끝 줄바꿈 없음) |
| mskc0002~mskc0004, mskc0008 | document | `Product Design & Quality` | .ocr | 마스커 사례 #44 T2~T4와 추가 사례 (본문은 사례 문장 하나, 끝 줄바꿈 없음) |
| copa0001 | document | `Marketing` | .ocr | 사본 문서 A (#23) |
| copa0002 | document | `Marketing;Government & Public Affairs` | .ocr | A의 사본 A′: 본문에 공백만 더함 (#23) |
| copb0001 | document | `Marketing` | .ocr | A와 한 글자 다른 문서 B: `colour` → `color` (#23) |
| vera0001 | document | `Marketing` | .ocr | 다른 판본이 있는 정답 문서 A (#50) |
| vera0002 | document | `Marketing;Government & Public Affairs` | .ocr | A의 다른 판본 A2: 마지막 문장만 다름 (#50) |
| verb0001 | document | `Marketing` | .ocr | A와 관계없는 문서 B (#50) |
| verc0001 | document | `Government & Public Affairs` | .ocr | 다중 홉 사례의 두 번째 정답 문서 C (#50) |
| hyba0001 | document | `Marketing` | .ocr | 하이브리드 검색의 정답 R: 질문과 드문 고유명 `Zorvex`만 겹침 (#52) |
| hybb0001, hybb0002 | document | `Marketing` | .ocr | 질문과 기능어(`what`, `does`, `have` 등)만 많이 겹치는 문서: 가짜 dense 임베딩에서 R보다 가깝다 (#52) |
| hybc0001 | document | `Government & Public Affairs` | .ocr | `Zorvex`가 든 regulatory 전용 문서: ACL 사례 (#52) |
| rrfb0001~rrfb0004 | document | `Marketing` | .ocr | RRF 동점 사례: 질문과 기능어만 겹쳐 가짜 dense에서 상위, BM25 점수 없음 (#59) |
| rrfy0001 | document | `Marketing` | .ocr | RRF 동점 사례: 드문 단어 `zorvex`가 들어 BM25 1위, dense 후보 밖 (#59) |
| rrfa0001 | document | `Marketing` | .ocr | RRF 사례: 드문 단어 `Plumbic`이 들어 dense·BM25 모두 1위 (#59) |
| rrfc0001 | document | `Government & Public Affairs` | .ocr | RRF 사례: `zorvex`가 든 regulatory 전용 문서, ACL 사례 (#59) |
