#!/usr/bin/env bash
#
# Launches every rank of a StakMesh cluster from a single terminal on one
# machine: the local rank runs directly, remote ranks run over ssh, and
# every rank's output streams live into this terminal, tagged and color
# coded per rank.
#
# Single source of truth for cluster membership: this file does NOT store
# hostnames. Rank -> host comes from the same topology file the actual
# mnist_distributed binary parses (configs/*.txt, see cluster_config.hpp),
# looked up by rank. This file only adds what the binary's config can't
# know: which ranks are local vs ssh, the remote build directory, and the
# binary filename per rank (mnist_distributed vs mnist_distributed.exe).
#
# Setup:
#   1. Copy scripts/cluster_nodes.conf.example to scripts/cluster_nodes.conf
#   2. Edit it: point `topology` at your real configs/*.txt file, and add
#      one line per rank (see comments in that file for the exact format)
#   3. Run this script from anywhere; extra args after `--` are appended to
#      every rank's command, e.g.:
#
#        ./scripts/launch_cluster.sh -- --epochs 5 --batch-size 512
#
# Adding/refreshing a remote node without a manual clone+build there:
#   Add a line "<rank> deploy <local-binary-path>", then pass --deploy as
#   the first argument. Before any rank starts, this scp's <local-binary-path>
#   (e.g. your own build/mnist_distributed, or a binary downloaded from a CI
#   run - see .github/workflows/build-binaries.yml) to that rank's remote
#   directory (from its own line above, so the destination is never
#   repeated by hand) on that rank's ssh target:
#
#     ./scripts/launch_cluster.sh --deploy -- --epochs 5
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
NODES_FILE="$SCRIPT_DIR/cluster_nodes.conf"

if [[ ! -f "$NODES_FILE" ]]; then
    echo "error: $NODES_FILE not found. Copy cluster_nodes.conf.example and fill it in." >&2
    exit 1
fi

# --deploy (must come before "--", if present) scp's binaries out to remote
# ranks per any "deploy" lines in the config, before launching anything.
DO_DEPLOY=0
if [[ "${1:-}" == "--deploy" ]]; then
    DO_DEPLOY=1
    shift
fi

# Everything after a literal "--" is forwarded to every rank's command.
EXTRA_ARGS=""
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS="$*"
fi

COLORS=($'\033[36m' $'\033[35m' $'\033[33m' $'\033[32m' $'\033[34m' $'\033[31m')
RESET=$'\033[0m'
BOLD=$'\033[1m'
GREEN=$'\033[32m'
RED=$'\033[31m'

BOX_WIDTH=60

# Prints "── label ──────..." padded to a consistent total width, so every
# section divider lines up the same regardless of label length.
section() {
    local label=" $1 "
    local dashes=$(( BOX_WIDTH - ${#label} - 2 ))
    (( dashes < 0 )) && dashes=0
    printf -- "──%s%s\n" "$label" "$(printf -- '─%.0s' $(seq 1 "$dashes"))"
}

# Every rank's raw (uncolored) output is also teed to its own log file, in
# addition to the live tagged/colored terminal view above - so a crash or
# hang on a remote rank leaves a trail behind even if you weren't watching
# the terminal at the time. Overwritten each run; grab a copy first if you
# want to keep it.
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

PIDS=()
RANKS_ORDER=()   # parallel array: RANKS_ORDER[i] is the rank that owns PIDS[i]

# Ctrl+C (or a TERM) is the only case that should print a "stopping" message -
# a normal, successful finish falls through to the run summary below instead
# and shouldn't look like an abort.
handle_signal() {
    echo ""
    section "stopping all ranks"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    section "all ranks stopped"
    exit 130
}
trap handle_signal INT TERM

echo "$(printf -- '═%.0s' $(seq 1 $BOX_WIDTH))"
echo "  ${BOLD}StakMesh cluster launcher${RESET}"
echo "$(printf -- '═%.0s' $(seq 1 $BOX_WIDTH))"

# ── Pass 1: read the "topology" directive and every rank's base line ───────
TOPOLOGY_REL=""       # e.g. "../configs/two_laptop_cluster.local.txt" - passed
                       # verbatim as --config on every rank's command
declare -A RANK_MODE   # rank -> local | ssh
declare -A RANK_USER   # rank -> ssh user (ssh ranks only)
declare -A RANK_DIR    # rank -> remote/local build directory
declare -A RANK_BIN    # rank -> binary filename

while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue

    if [[ "$line" == topology* ]]; then
        read -r _ TOPOLOGY_REL <<< "$line"
        continue
    fi

    read -r rank mode rest <<< "$line"
    case "$mode" in
        local)
            read -r dir bin <<< "$rest"
            RANK_MODE["$rank"]="local"
            RANK_DIR["$rank"]="$dir"
            RANK_BIN["$rank"]="$bin"
            ;;
        ssh)
            read -r user dir bin <<< "$rest"
            RANK_MODE["$rank"]="ssh"
            RANK_USER["$rank"]="$user"
            RANK_DIR["$rank"]="$dir"
            RANK_BIN["$rank"]="$bin"
            ;;
        deploy) : ;;  # handled in pass 2, below
        *)
            echo "warning: unknown mode '$mode' on line: $line (expected 'local', 'ssh', or 'deploy')" >&2
            ;;
    esac
done < "$NODES_FILE"

if [[ -z "$TOPOLOGY_REL" ]]; then
    echo "error: no 'topology <relative-path>' line found in $NODES_FILE" >&2
    exit 1
fi

# Resolve the topology file locally (relative to the repo root) purely to
# look up rank -> host below. TOPOLOGY_REL itself, unresolved, is what gets
# passed as --config on every rank's command, since it's relative to each
# rank's own build directory, not to this script.
TOPOLOGY_LOCAL="$REPO_ROOT/configs/$(basename "$TOPOLOGY_REL")"
if [[ ! -f "$TOPOLOGY_LOCAL" ]]; then
    echo "error: topology file not found at $TOPOLOGY_LOCAL" >&2
    exit 1
fi

declare -A RANK_HOST
while read -r trank thost tport; do
    [[ -z "$trank" ]] && continue
    RANK_HOST["$trank"]="$thost"
done < <(sed 's/#.*//' "$TOPOLOGY_LOCAL" | awk 'NF>=3 {print $1, $2, $3}')

# ── Optional deploy pass: scp each rank's binary to its own remote dir ─────
if [[ "$DO_DEPLOY" -eq 1 ]]; then
    section "deploying"
    deployed_any=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        read -r rank mode rest <<< "$line"
        [[ "$mode" != "deploy" ]] && continue
        deployed_any=1

        local_path="$rest"
        rank_mode="${RANK_MODE[$rank]:-}"
        if [[ "$rank_mode" != "ssh" ]]; then
            echo "error: deploy line for rank $rank has no matching 'ssh' line for that rank" >&2
            exit 1
        fi
        host="${RANK_HOST[$rank]:-}"
        if [[ -z "$host" ]]; then
            echo "error: rank $rank has no host entry in $TOPOLOGY_LOCAL" >&2
            exit 1
        fi
        if [[ ! -f "$local_path" ]]; then
            echo "error: deploy source '$local_path' for rank $rank not found" >&2
            exit 1
        fi

        dir="${RANK_DIR[$rank]}"
        bin="${RANK_BIN[$rank]}"
        if [[ "$dir" == *'\'* ]]; then
            remote_path="${dir}\\${bin}"   # windows-style remote dir
        else
            remote_path="${dir}/${bin}"    # posix-style remote dir
        fi
        target="${RANK_USER[$rank]}@${host}"

        echo "  rank${rank}  $(basename "$local_path") → ${target}"
        if ! scp -q "$local_path" "${target}:${remote_path}"; then
            echo "error: scp to ${target} failed - is it reachable over the tailnet?" >&2
            exit 1
        fi
    done < "$NODES_FILE"
    if [[ "$deployed_any" -eq 0 ]]; then
        echo "warning: --deploy was passed but no 'deploy' lines found in $NODES_FILE" >&2
    fi
fi

# ── Pass 2: launch every rank ───────────────────────────────────────────────
section "launching ${#RANK_MODE[@]} rank(s)"
i=0
for rank in $(printf '%s\n' "${!RANK_MODE[@]}" | sort -n); do
    mode="${RANK_MODE[$rank]}"
    dir="${RANK_DIR[$rank]}"
    bin="${RANK_BIN[$rank]}"
    color="${COLORS[$((i % ${#COLORS[@]}))]}"
    i=$((i + 1))

    if [[ "$dir" == *'\'* ]]; then
        bin_invoke="$bin"                     # windows: no ./ prefix needed
    else
        bin_invoke="./$bin"                   # posix
    fi
    cmd="cd ${dir} && ${bin_invoke} --config ${TOPOLOGY_REL} ${EXTRA_ARGS}"

    log_file="$LOG_DIR/rank${rank}.log"

    if [[ "$mode" == "local" ]]; then
        echo "  rank${rank}  local     ${cmd}"
        ( eval "$cmd" 2>&1 | tee "$log_file" | sed -u "s/^/${color}[rank${rank}]${RESET} /" ) &
        PIDS+=($!)
        RANKS_ORDER+=("$rank")
    else
        host="${RANK_HOST[$rank]:-}"
        if [[ -z "$host" ]]; then
            echo "error: rank $rank has no host entry in $TOPOLOGY_LOCAL" >&2
            exit 1
        fi
        target="${RANK_USER[$rank]}@${host}"
        echo "  rank${rank}  ssh:${target}   ${cmd}"
        ( ssh "$target" "$cmd" 2>&1 | tee "$log_file" | sed -u "s/^/${color}[rank${rank}]${RESET} /" ) &
        PIDS+=($!)
        RANKS_ORDER+=("$rank")
    fi
done

section "running"
echo "  logs → ${LOG_DIR}/rank<N>.log   (Ctrl+C to stop all ranks)"
echo

STATUS=()
for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
        STATUS[$idx]=0
    else
        STATUS[$idx]=$?
    fi
done

echo ""
section "run summary"
overall=0
for idx in "${!PIDS[@]}"; do
    rank="${RANKS_ORDER[$idx]}"
    if [[ "${STATUS[$idx]}" -eq 0 ]]; then
        echo "  ${GREEN}✓${RESET} rank${rank}"
    else
        echo "  ${RED}✗${RESET} rank${rank}  (exit ${STATUS[$idx]}) → logs/rank${rank}.log"
        overall=1
    fi
done
echo "$(printf -- '─%.0s' $(seq 1 $BOX_WIDTH))"
exit "$overall"