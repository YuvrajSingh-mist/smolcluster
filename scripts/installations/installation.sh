#!/usr/bin/env bash
set -euo pipefail

log() {
    echo "[install] $*"
}

warn() {
    echo "[install][warn] $*"
}

err() {
    echo "[install][error] $*" >&2
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Required command not found: $1"
        return 1
    fi
}

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        log "uv already installed: $(command -v uv)"
        return 0
    fi

    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh |

    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        warn "uv installed but not found in currentell PATH."
        warn "Add this to yourell config: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    else
        log "uv installed: $(command -v uv)"
    fi
}

install_macos() {
    log "Detected macOS"

    if ! command -v brew >/dev/null 2>&1; then
        err "Homebrew is required on macOS. Install from https://brew.sh and rerun."
        exit 1
    fi

    log "Updating Homebrew..."
    brew update

    log "Installing tmux, docker, colima, redis..."
    brew install tmux docker colima redis

    if ! colima status >/dev/null 2>&1; then
        log "Starting Colima..."
        colima start
    else
        log "Colima already running"
    fi

    log "Verifying Docker daemon via Colima"
    docker ps

    if command -v redis-cli >/dev/null 2>&1; then
        log "redis-cli available: $(command -v redis-cli)"
    fi

    install_uv

    log "macOS dependency setup complete"
}

install_linux() {
    log "Detected Linux"

    require_cmd sudo
    require_cmd apt
    require_cmd systemctl

    log "Installing Docker, tmux, curl, redis-tools..."
    sudo apt update
    sudo apt install -y docker.io tmux curl ca-certificates redis-tools

    log "Enabling and starting Docker service"
    sudo systemctl enable docker
    sudo systemctl start docker

    if id -nG "$USER" | grep -qw docker; then
        log "User already in docker group"
    else
        log "Adding user to docker group"
        sudo usermod -aG docker "$USER"
        warn "Group change requires re-login or: newgrp docker"
    fi

    # Jetson / NVIDIA runtime path
    if [[ -f /etc/nv_tegra_release ]] || [[ "$(uname -m)" == "aarch64" ]]; then
        log "Jetson/ARM environment detected; attempting NVIDIA container toolkit install"
        if sudo apt install -y nvidia-container-toolkit; then
            sudo systemctl restart docker
            log "nvidia-container-toolkit installed"
        else
            warn "Could not install nvidia-container-toolkit automatically."
            warn "Install NVIDIA runtime manually if GPU containers are required."
        fi
    fi

    install_uv

    log "Checking Docker"
    if docker ps >/dev/null 2>&1; then
        docker ps
    else
        warn "docker ps failed in currentell (likely needs re-login/newgrp)."
        warn "Try: newgrp docker"
        warn "Or use: sudo docker ps"
    fi

    if command -v redis-cli >/dev/null 2>&1; then
        log "redis-cli available: $(command -v redis-cli)"
    fi

    log "Linux dependency setup complete"
}

main() {
    require_cmd curl

    case "$(uname -s)" in
        Darwin)
            install_macos
            ;;
        Linux)
            install_linux
            ;;
        *)
            err "Unsupported OS: $(uname -s)"
            exit 1
            ;;
    esac

    log "Done"
}

main "$@"
