#!/usr/bin/env bash
#
# Single-machine StakMesh check: launches N ranks of stakmesh_smoke as
# SEPARATE PROCESSES talking over real TCP sockets on 127.0.0.1.
#
# This exercises the same code path as a Tailscale/LAN run - same sockets,
# same ring bootstrap, same collectives, same framing - minus the network
# itself. A pass here means the logic is sound; it cannot catch firewall,
# routing, MTU or latency problems (run the same binary over --config for that).
#
# Usage:
#   scripts/loopback_test.sh [-n WORLD_SIZE] [-b path/to/stakmesh_smoke]
#                            [-t TIMEOUT_SEC] [--peer-death] [-- extra smoke args]
#
#   -n N            number of ranks (default 2)
#   -b BIN          smoke binary (default: build/stakmesh_smoke, then ./stakmesh_smoke)
#   -t SEC          hard per-run timeout, a hang counts as failure (default 120)
#   --peer-death    fault test: last rank dies abruptly mid-training; the
#                   survivors must exit with a clean comm error (code 3),
#                   not hang and not get killed by a signal (SIGPIPE).
#
# Env: STAKMESH_TEST_BASE_PORT pins the first port (default: random free range).
#      OMP_NUM_THREADS defaults to 1 here (override by exporting it). All ranks
#      share this machine's cores, and StakML's OpenMP matmul would otherwise
#      start one spinning thread per core in EVERY rank; on a laptop that made
#      the 2-rank run take ~11 s instead of ~0.3 s. A real multi-machine run
#      has its own cores per rank, so this only applies to loopback.
set -uo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

WORLD=2
BIN=""
TIMEOUT=120
PEER_DEATH=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n) WORLD="$2"; shift 2 ;;
        -b) BIN="$2"; shift 2 ;;
        -t) TIMEOUT="$2"; shift 2 ;;
        --peer-death) PEER_DEATH=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n "2,30p" "$0"; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
done

if [[ -z "$BIN" ]]; then
    for c in "$REPO_ROOT/build/stakmesh_smoke" "$REPO_ROOT/build/stakmesh_smoke.exe" ./stakmesh_smoke; do
        [[ -x "$c" ]] && BIN="$c" && break
    done
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
    echo "error: stakmesh_smoke binary not found. Build first: cmake --build build" >&2
    exit 2
fi
if (( WORLD < 2 )); then echo "error: -n must be >= 2" >&2; exit 2; fi

port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

# Find WORLD consecutive free ports.
pick_base_port() {
    local base tries=0
    while (( tries < 50 )); do
        if [[ -n "${STAKMESH_TEST_BASE_PORT:-}" ]]; then base="$STAKMESH_TEST_BASE_PORT"
        else base=$(( 30000 + (RANDOM * 7 + RANDOM) % 28000 )); fi
        local ok=1 i
        for (( i = 0; i < WORLD; i++ )); do port_in_use $(( base + i )) && ok=0; done
        (( ok )) && { echo "$base"; return 0; }
        [[ -n "${STAKMESH_TEST_BASE_PORT:-}" ]] && break
        tries=$(( tries + 1 ))
    done
    return 1
}

BASE_PORT="$(pick_base_port)" || { echo "error: no free port range found" >&2; exit 2; }
PEERS=""
for (( i = 0; i < WORLD; i++ )); do PEERS+="${PEERS:+,}127.0.0.1:$(( BASE_PORT + i ))"; done

LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/stakmesh_loopback.XXXXXX")"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

echo "StakMesh loopback test: $WORLD ranks, ports $BASE_PORT..$(( BASE_PORT + WORLD - 1 )), timeout ${TIMEOUT}s"
echo "  binary: $BIN"
(( PEER_DEATH )) && echo "  mode:   peer-death (rank $(( WORLD - 1 )) will exit abruptly mid-training)"

SMOKE_ARGS=(--peers "$PEERS")
(( PEER_DEATH )) && SMOKE_ARGS+=(--die-at-step 20 --io-timeout-ms 8000)

START=$SECONDS
for (( r = 0; r < WORLD; r++ )); do
    "$BIN" --rank "$r" "${SMOKE_ARGS[@]}" "${EXTRA[@]}" >"$LOG_DIR/rank$r.log" 2>&1 &
    PIDS+=($!)
done

# Wait for every rank, with a hard deadline: a hang is a failure, not a stuck CI job.
CODES=()
for (( r = 0; r < WORLD; r++ )); do
    while kill -0 "${PIDS[$r]}" 2>/dev/null; do
        if (( SECONDS - START >= TIMEOUT )); then
            echo "TIMEOUT: rank $r still running after ${TIMEOUT}s - treating as a hang" >&2
            cleanup
            break
        fi
        sleep 0.2
    done
    wait "${PIDS[$r]}" 2>/dev/null; CODES+=($?)
done

for (( r = 0; r < WORLD; r++ )); do
    echo "──── rank $r (exit ${CODES[$r]}) ────"
    cat "$LOG_DIR/rank$r.log"
done
echo "Total: $(( SECONDS - START ))s   (logs kept in $LOG_DIR)"

FAIL=0
if (( PEER_DEATH )); then
    # Last rank exits 0 via _Exit. Everyone else must report a clean comm error (3).
    # Anything >= 128 means a signal killed the process (e.g. SIGPIPE = 141).
    for (( r = 0; r < WORLD; r++ )); do
        want=3; (( r == WORLD - 1 )) && want=0
        if (( CODES[r] != want )); then
            echo "FAIL: rank $r exited ${CODES[$r]}, expected $want" >&2; FAIL=1
        fi
    done
    if (( FAIL == 0 )); then echo "PASS: survivors failed cleanly after peer death"; fi
else
    for (( r = 0; r < WORLD; r++ )); do
        if (( CODES[r] != 0 )); then echo "FAIL: rank $r exited ${CODES[$r]}" >&2; FAIL=1; fi
    done
    if (( FAIL == 0 )); then echo "PASS: all $WORLD ranks passed"; fi
fi
exit $FAIL
