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
