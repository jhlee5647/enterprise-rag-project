#!/usr/bin/env bash
# Pod 안에서 Qdrant 서버를 띄우고 멈춘다. 색인 위치(config의 paths.index)가 http://localhost:6333일 때 쓴다.
#
#   scripts/qdrant-server.sh start    # 처음이면 서버를 내려받고, 127.0.0.1:6333으로 띄운다
#   scripts/qdrant-server.sh status   # 떠 있는지, 컬렉션 목록
#   scripts/qdrant-server.sh stop
#
# 서버 버전은 qdrant-client(1.19.1)와 같은 마이너 버전이다. 바이너리·저장소·로그는 Pod를 다시 시작해도
# 남도록 /workspace 볼륨 아래(QDRANT_HOME, 기본 /workspace/qdrant)에 둔다.
set -euo pipefail

VERSION=v1.19.2
HOME_DIR=${QDRANT_HOME:-/workspace/qdrant}
BIN=$HOME_DIR/bin/qdrant
PID_FILE=$HOME_DIR/qdrant.pid
LOG=$HOME_DIR/qdrant.log
URL=http://127.0.0.1:6333

# pid 파일의 프로세스가 살아 있고 qdrant인지 (pid가 다른 프로세스에 다시 쓰였을 수 있다)
running() {
  [[ -f $PID_FILE ]] && grep -qa qdrant "/proc/$(cat "$PID_FILE")/cmdline" 2>/dev/null
}

download_server() {
  [[ -x $BIN ]] && return
  mkdir -p "$HOME_DIR/bin"
  echo "Qdrant $VERSION 내려받는 중"
  # 받다가 끊겨도 반쪽짜리 실행 파일이 남지 않도록 임시 폴더에 풀고 옮긴다
  local tmp
  tmp=$(mktemp -d "$HOME_DIR/download.XXXXXX")
  curl -fsSL "https://github.com/qdrant/qdrant/releases/download/$VERSION/qdrant-x86_64-unknown-linux-musl.tar.gz" \
    | tar -xz --no-same-owner -C "$tmp" qdrant
  mv "$tmp/qdrant" "$BIN"
  rmdir "$tmp"
}

case ${1:-} in
  start)
    if running; then echo "이미 떠 있다: $URL (pid $(cat "$PID_FILE"))"; exit 0; fi
    download_server
    mkdir -p "$HOME_DIR/storage" "$HOME_DIR/snapshots"
    QDRANT__SERVICE__HOST=127.0.0.1 \
    QDRANT__STORAGE__STORAGE_PATH=$HOME_DIR/storage \
    QDRANT__STORAGE__SNAPSHOTS_PATH=$HOME_DIR/snapshots \
    QDRANT__TELEMETRY_DISABLED=true \
      nohup "$BIN" >>"$LOG" 2>&1 &
    echo $! >"$PID_FILE"
    for _ in $(seq 60); do
      if curl -fs "$URL/readyz" >/dev/null; then echo "떠 있다: $URL (로그 $LOG)"; exit 0; fi
      if ! running; then rm -f "$PID_FILE"; echo "서버가 바로 멈췄다. 로그: $LOG" >&2; exit 1; fi
      sleep 1
    done
    kill "$(cat "$PID_FILE")"; rm -f "$PID_FILE"
    echo "60초 안에 준비되지 않아 멈췄다. 로그: $LOG" >&2; exit 1
    ;;
  stop)
    if ! running; then echo "떠 있지 않다"; exit 0; fi
    kill "$(cat "$PID_FILE")"
    for _ in $(seq 30); do running || break; sleep 1; done
    if running; then kill -9 "$(cat "$PID_FILE")"; echo "30초 안에 멈추지 않아 강제로 멈췄다"; fi
    rm -f "$PID_FILE"; echo "멈췄다"
    ;;
  status)
    if running; then echo "떠 있다: $URL (pid $(cat "$PID_FILE"))"; curl -fs "$URL/collections"; echo
    else echo "떠 있지 않다"; exit 1; fi
    ;;
  *)
    echo "사용: $0 start|stop|status" >&2; exit 2
    ;;
esac
