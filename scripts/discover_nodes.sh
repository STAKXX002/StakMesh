#!/usr/bin/env bash
#
# Lists Tailscale peers that look like StakMesh cluster candidates: hostname
# matches the given prefix, currently online, and running an OS this project
# actually trains on (linux/windows). A machine that's asleep, powered off,
# or dual-booted into an OS it's not currently running (e.g. a Windows box
# rebooted into Linux) simply won't show as an online Tailscale peer for
# that OS, so it's excluded here automatically - no per-machine flag needed.
#
# Pure bash + awk, no jq or python: parses the plain-text `tailscale status`
# table directly instead of --json.
#
# Usage:
#   ./scripts/discover_nodes.sh                  # prefix defaults to stakxx002-
#   ./scripts/discover_nodes.sh myprefix-
#
set -euo pipefail

PREFIX="${1:-stakxx002-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODES_FILE="$SCRIPT_DIR/cluster_nodes.conf"

if ! command -v tailscale >/dev/null 2>&1; then
    echo "error: 'tailscale' not found on PATH" >&2
    exit 1
fi

# Used to recognize this machine's own line in `tailscale status` and skip
# it (you don't deploy to yourself, rank 0 is already local).
SELF_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"

[[ -f "$NODES_FILE" ]] && CONFIGURED="$(cat "$NODES_FILE")" || CONFIGURED=""

online=()
offline=()
wrong_os=()

# tailscale status columns: IP  HOSTNAME  USER  OS  STATUS...
while read -r ip host os status_rest; do
    [[ "$host" != "$PREFIX"* ]] && continue
    [[ "$ip" == "$SELF_IP" ]] && continue

    if [[ "$os" != "linux" && "$os" != "windows" ]]; then
        wrong_os+=("$host	$os")
    elif [[ "$status_rest" == offline* ]]; then
        offline+=("$host	$os")
    else
        online+=("$host	$os	$ip")
    fi
done < <(tailscale status | awk '{ ip=$1; host=$2; os=$4; rest=""; for (i=5;i<=NF;i++) rest = rest (i>5?" ":"") $i; print ip, host, os, rest }')

echo "This machine's Tailscale IP: ${SELF_IP:-unknown}"
echo

echo "Online, trainer-capable (${PREFIX}*):"
if [[ ${#online[@]} -eq 0 ]]; then
    echo "  (none)"
else
    for entry in "${online[@]}"; do
        IFS=$'\t' read -r host os ip <<< "$entry"
        if [[ "$CONFIGURED" == *"$host"* ]]; then
            tag="already in cluster_nodes.conf"
        else
            tag="NOT yet configured"
        fi
        printf "  %-20s %-8s %-16s %s\n" "$host" "$os" "$ip" "$tag"
    done
fi

echo
echo "Skipped, offline or asleep:"
if [[ ${#offline[@]} -eq 0 ]]; then
    echo "  (none)"
else
    for entry in "${offline[@]}"; do
        IFS=$'\t' read -r host os <<< "$entry"
        printf "  %-20s %-8s\n" "$host" "$os"
    done
fi

if [[ ${#wrong_os[@]} -gt 0 ]]; then
    echo
    echo "Skipped, non-trainer OS:"
    for entry in "${wrong_os[@]}"; do
        IFS=$'\t' read -r host os <<< "$entry"
        printf "  %-20s %-8s\n" "$host" "$os"
    done
fi