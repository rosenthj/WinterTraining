#!/bin/bash
#
# Push the parts of this repository that git cannot deliver to the cluster checkout.
#
# Code arrives on the cluster by `git pull`. What git deliberately does not track -- the
# fastchess binary and the DFRC opening book -- has to get there some other way, and that
# is what this script is for. It can also push tracked source with --code, for trying an
# uncommitted change without committing it first.
#
# The Winter engine binary is NOT synced: a local build does not run on the cluster and
# must be compiled there. Point ENGINE at it in datagen_config.sh, or build it as
# datagen/Winter.
#
# Usage:
#   ./sync_to_cluster.sh [options] [extra paths...]
#
# Options:
#   -n, --dry-run     list what would transfer, change nothing
#   -c, --code        also push tracked source (src/, datagen/*.sh, *.sh, README.md)
#       --no-assets   skip the binaries and opening book
#       --env         also push local_env.sh (see the warning below)
#   -h, --help        this message
#
# Any extra arguments are treated as additional paths to push, relative to the repository
# root, so a one-off file needs no flag:
#
#   ./sync_to_cluster.sh src/merge_datasets.py
#
# The destination is WINTER_SYNC_DEST, set in the git-ignored local_env.sh so that no
# cluster path is committed. Either an rsync remote or a plain path when running on the
# cluster itself:
#
#   WINTER_SYNC_DEST="user@host:/scratch/.../git/WinterTraining"
#   WINTER_SYNC_DEST="/scratch/.../git/WinterTraining"
#
# --env is off by default on purpose: the cluster's local_env.sh holds *its* paths, which
# are not the ones on your workstation, and pushing yours would overwrite them. Create it
# once on the cluster from local_env.sh.example instead.
#
# Nothing is ever deleted at the destination: rsync runs without --delete, so this only
# adds and updates. .git/ is always excluded, so the cluster checkout's own history and
# working state are never touched.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$REPO_DIR/local_env.sh" ]; then
    # shellcheck source=/dev/null
    source "$REPO_DIR/local_env.sh"
fi

DRY_RUN=0
WITH_CODE=0
WITH_ASSETS=1
WITH_ENV=0
EXTRA=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -c|--code)    WITH_CODE=1 ;;
        --no-assets)  WITH_ASSETS=0 ;;
        --env)        WITH_ENV=1 ;;
        -h|--help)    sed -n '2,42p' "$0" | sed 's/^#\{0,1\} \{0,1\}//'; exit 0 ;;
        -*)           echo "Unknown option: $1" >&2; exit 1 ;;
        *)            EXTRA+=( "$1" ) ;;
    esac
    shift
done

if [ -z "${WINTER_SYNC_DEST:-}" ]; then
    echo "Error: WINTER_SYNC_DEST is not set." >&2
    echo "       Add it to local_env.sh (copy local_env.sh.example if you have none):" >&2
    echo "         WINTER_SYNC_DEST=\"user@host:/path/to/git/WinterTraining\"" >&2
    exit 1
fi

RSYNC=( rsync -rlptv --human-readable --exclude '.git/' --exclude '__pycache__/' )
[ "$DRY_RUN" -eq 1 ] && RSYNC+=( --dry-run )
# Compression helps over the network and costs little locally.
case "$WINTER_SYNC_DEST" in
    *:*) RSYNC+=( --compress --partial ) ;;
esac

# Build the source list. Paths are relative to the repository root and pushed with their
# directory structure intact, so rsync is invoked from there with --relative.
SOURCES=()

if [ "$WITH_ASSETS" -eq 1 ]; then
    # Deliberately excludes the Winter engine binary: a locally built one does not run on
    # the cluster, so it has to be compiled there. Pushing ours would replace a working
    # build with a broken one. fastchess is static and portable, so it does travel.
    for asset in datagen/fastchess datagen/DFRC.epd; do
        [ -e "$REPO_DIR/$asset" ] && SOURCES+=( "$asset" )
    done
fi

if [ "$WITH_CODE" -eq 1 ]; then
    SOURCES+=( src datagen README.md )
    for script in "$REPO_DIR"/*.sh; do
        [ -e "$script" ] || continue
        # local_env.sh is machine-specific and must not ride along with the code: the
        # cluster's copy holds *its* paths. --env is the only way to push it.
        [ "$(basename "$script")" = "local_env.sh" ] && continue
        SOURCES+=( "$(basename "$script")" )
    done
fi

if [ "$WITH_ENV" -eq 1 ]; then
    if [ -f "$REPO_DIR/local_env.sh" ]; then
        SOURCES+=( local_env.sh )
    else
        echo "Warning: --env given but local_env.sh does not exist here." >&2
    fi
fi

SOURCES+=( "${EXTRA[@]}" )

if [ "${#SOURCES[@]}" -eq 0 ]; then
    echo "Nothing to sync. Use --code, drop --no-assets, or name paths explicitly." >&2
    exit 1
fi

for src in "${SOURCES[@]}"; do
    if [ ! -e "$REPO_DIR/$src" ]; then
        echo "Error: no such path in the repository: $src" >&2
        exit 1
    fi
done

# Generated data and checkpoints are large and belong on the cluster only; never push them
# even when a directory that contains them is named.
RSYNC+=( --exclude 'data/' --exclude 'logs/' --exclude '*.npz' --exclude '*.pt'
         --exclude '*.pgn' --exclude '.gitignore' )
# The engine must be compiled on the cluster, so a local build must never reach it -- not
# via the asset list, and not by riding along inside datagen/ under --code. Excluding it
# here also protects a working cluster build from being overwritten.
RSYNC+=( --exclude 'Winter' --exclude 'Winter_*' )

echo "Syncing to $WINTER_SYNC_DEST"
[ "$DRY_RUN" -eq 1 ] && echo "(dry run -- nothing will be written)"
printf '  %s\n' "${SOURCES[@]}"
echo

cd "$REPO_DIR"
"${RSYNC[@]}" --relative "${SOURCES[@]}" "$WINTER_SYNC_DEST/"

echo
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run complete. Rerun without -n to transfer."
else
    echo "Done. Code itself is delivered by git; run 'git pull' in the cluster checkout."
fi
