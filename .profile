# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$AFS/.bashrc.lfs" ]; then
        . "$AFS/.bashrc.lfs"
    fi
fi
