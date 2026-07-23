#!/usr/bin/env bash
#
# StakMesh cluster launcher.
#
# Starts every rank in the cluster from ONE terminal on this (the control)
# machine: the local rank runs directly, remote ranks are started over SSH.
# All output is tagged with "[rank N]" and color-coded per rank, streamed
# live and interleaved. Ctrl+C stops every rank, local and remote.
#
# Setup:
#   1. Copy scripts/cluster_nodes.conf.example to scripts/cluster_nodes.conf
#   2. Edit it to describe your actual machines (see comments in that file)
#   3. Run this script from anywhere; extra args after `--` are appended to
#      every rank's command, e.g.:
#
#        ./scripts/launch_cluster.sh -- --epochs 5 --batch-size 512
#
# Adding/refreshing a remote node without a manual clone+build there:
#   Add a line to cluster_nodes.conf, rank first like every other line:
#
#     <rank> deploy <local-binary-path> <remote-dest-path>
#
#   then pass --deploy as the first argument. Before any rank starts, this
#   scp's <local-binary-path> (e.g. your own build/mnist_distributed, or a
#   binary downloaded from a CI run - see .github/workflows/build-binaries.yml)
#   to <remote-dest-path> on that rank's ssh target. Build once on whichever
#   machine, deploy everywhere else on the tailnet:
#
#     ./scripts/launch_cluster.sh --deploy -- --epochs 5
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES_FILE="$SCRIPT_DIR/cluster_nodes.conf"

if [[ ! -f "$NODES_FILE" ]]; then
    echo "error: $NODES_FILE not found." >&2
    echo "  Copy scripts/cluster_nodes.conf.example to scripts/cluster_nodes.conf" >&2
    echo "  and fill in your actual machines first." >&2
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

COLORS=(31 32 33 34 35 36 91 92)  # red green yellow blue magenta cyan + bright variants, cycles if >8 ranks
PIDS=()
LABELS=()

CLEANUP_DONE=0
cleanup() {
    [[ "$CLEANUP_DONE" -eq 1 ]] && return
    CLEANUP_DONE=1
    echo
    echo "── stopping all ranks ──────────────────────────────────"
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    echo "── all ranks stopped ───────────────────────────────────"
}
trap cleanup INT TERM EXIT

echo "══════════════════════════════════════════════════════════"
echo "  StakMesh cluster launcher"
echo "══════════════════════════════════════════════════════════"

# First pass: rank -> ssh target, so "deploy" lines (which only name a rank,
# not a host - the host already lives on that rank's ssh line) know where to
# scp to. Only ssh-mode ranks need this; a "local" rank deploys nowhere.
declare -A RANK_TARGET
while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue
    read -r rank mode rest <<< "$line"
    if [[ "$mode" == "ssh" ]]; then
        read -r target _ <<< "$rest"
        RANK_TARGET["$rank"]="$target"
    fi
done < "$NODES_FILE"

if [[ "$DO_DEPLOY" -eq 1 ]]; then
    echo "── deploying binaries ──────────────────────────────────"
    deployed_any=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line//[[:space:]]/}" ]] && continue
        read -r rank mode rest <<< "$line"
        [[ "$mode" != "deploy" ]] && continue
        deployed_any=1

        read -r local_path remote_path <<< "$rest"
        target="${RANK_TARGET[$rank]:-}"
        if [[ -z "$target" ]]; then
            echo "error: deploy line for rank $rank has no matching 'ssh' line for that rank" >&2
            exit 1
        fi
        if [[ ! -f "$local_path" ]]; then
            echo "error: deploy source '$local_path' for rank $rank not found" >&2
            exit 1
        fi

        echo "  rank${rank}: ${local_path} -> ${target}:${remote_path}"
        if ! scp -q "$local_path" "${target}:${remote_path}"; then
            echo "error: scp to ${target} failed - is it reachable over the tailnet?" >&2
            exit 1
        fi
    done < "$NODES_FILE"
    if [[ "$deployed_any" -eq 0 ]]; then
        echo "warning: --deploy was passed but no 'deploy' lines found in $NODES_FILE" >&2
    fi
    echo "── deploy complete ─────────────────────────────────────"
fi

i=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # skip comments and blank lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue

    read -r rank mode rest <<< "$line"
    [[ "$mode" == "deploy" ]] && continue  # handled in the deploy pass above, not a rank to launch
    color="${COLORS[$((i % ${#COLORS[@]}))]}"
    label="rank${rank}"
    prefix="$(printf '\033[1;%sm[%s]\033[0m' "$color" "$label")"
    i=$((i + 1))

    case "$mode" in
        local)
            cmd="$rest $EXTRA_ARGS"
            printf '%s\n' "${prefix} starting locally:"
            printf '%s\n' "${prefix}   $cmd"
            # NOTE: process substitution (not a `| sed` pipeline) so that $!
            # below is the PID of the actual command, not the sed formatter.
            # A `| pipe` backgrounds sed as the last stage and $! grabs
            # THAT pid - killing it does nothing to the real process.
            bash -lc "$cmd" > >(sed -u "s/^/${prefix} /") 2>&1 &
            PIDS+=("$!")
            ;;
        ssh)
            read -r target sshcmd <<< "$rest"
            cmd="$sshcmd $EXTRA_ARGS"
            printf '%s\n' "${prefix} starting on ${target} via ssh:"
            printf '%s\n' "${prefix}   $cmd"
            # -t allocates a pseudo-tty so the remote process gets SIGHUP
            # (and dies) when the ssh connection is closed on Ctrl+C.
            ssh -t "$target" "$cmd" > >(sed -u "s/^/${prefix} /") 2>&1 &
            PIDS+=("$!")
            ;;
        *)
            echo "warning: unknown mode '$mode' on line: $line (expected 'local' or 'ssh')" >&2
            ;;
    esac
done < "$NODES_FILE"

if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "error: no ranks started - check $NODES_FILE" >&2
    exit 1
fi

echo "── ${#PIDS[@]} rank(s) running, Ctrl+C to stop all ────────"
wait