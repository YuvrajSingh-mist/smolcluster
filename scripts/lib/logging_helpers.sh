#!/bin/bash

resolve_compose_cmd() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
        return 0
    fi

    if command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(docker-compose)
        return 0
    fi

    COMPOSE_CMD=()
    return 1
}

start_logging_stack() {
    local project_dir="$1"
    local compose_file="$project_dir/logging/docker-compose.yml"

    mkdir -p "$project_dir/logging/cluster-logs"

    echo ""
    echo "📈 Starting logging infrastructure on controller..."

    if [[ ! -f "$compose_file" ]]; then
        echo "⚠️  Logging not configured (logging/docker-compose.yml not found)"
        return 0
    fi

    if ! command -v docker >/dev/null 2>&1; then
        echo "⚠️  Docker CLI not found. Skipping centralized logging setup."
        return 0
    fi

    if ! docker info >/dev/null 2>&1; then
        echo "⚠️  Docker daemon not running. Skipping centralized logging setup."
        return 0
    fi

    if ! resolve_compose_cmd; then
        echo "⚠️  Neither 'docker compose' nor 'docker-compose' is available. Skipping centralized logging setup."
        return 0
    fi

    if docker ps --format '{{.Names}}' | grep -qx 'loki'; then
        echo "🧹 Cleaning up old logs from Loki..."
        (cd "$project_dir/logging" && "${COMPOSE_CMD[@]}" down loki && docker volume rm logging_loki-data || true)
        (cd "$project_dir/logging" && "${COMPOSE_CMD[@]}" up -d loki)
        sleep 3
        if curl -s http://localhost:3100/ready | grep -q "ready"; then
            echo "✅ Loki restarted with fresh database"
        else
            echo "⚠️  Loki may not be ready yet, but continuing..."
        fi

        if ! docker ps --format '{{.Names}}' | grep -qx 'grafana'; then
            (cd "$project_dir/logging" && "${COMPOSE_CMD[@]}" up -d grafana)
            echo "📊 Grafana UI at http://localhost:3000 (admin/admin)"
        fi
        return 0
    fi

    echo "🚀 Starting Loki + Grafana..."
    (cd "$project_dir/logging" && "${COMPOSE_CMD[@]}" up -d)
    sleep 3
    if curl -s http://localhost:3100/ready | grep -q "ready"; then
        echo "✅ Loki ready at http://localhost:3100"
        echo "📊 Grafana UI at http://localhost:3000 (admin/admin)"
    else
        echo "⚠️  Loki may not be ready yet, but continuing..."
    fi
}