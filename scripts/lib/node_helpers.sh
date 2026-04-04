#!/bin/bash

init_node_helpers() {
    NODE_HELPERS_CONFIG_FILE="$1"
    NODE_HELPERS_PROJECT_DIR="$2"
    NODE_HELPERS_REMOTE_PROJECT_DIR="$3"

    NODE_HELPERS_LOCAL_HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
    NODE_HELPERS_LOCAL_HOST_FULL="$(hostname -f 2>/dev/null || hostname)"
    NODE_HELPERS_LOCAL_IPS=()

    while IFS= read -r ip_addr; do
        [[ -n "$ip_addr" ]] || continue
        case " ${NODE_HELPERS_LOCAL_IPS[*]} " in
            *" $ip_addr "*) ;;
            *) NODE_HELPERS_LOCAL_IPS+=("$ip_addr") ;;
        esac
    done < <(
        {
            hostname -I 2>/dev/null | tr ' ' '\n'
            ip -o -4 addr show up scope global 2>/dev/null | awk '{split($4, parts, "/"); print parts[1]}'
        } | awk 'NF'
    )

    NODE_HELPERS_LOCAL_HOSTS=(
        "$NODE_HELPERS_LOCAL_HOST_SHORT"
        "$NODE_HELPERS_LOCAL_HOST_FULL"
    )
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        case " ${NODE_HELPERS_LOCAL_HOSTS[*]} " in
            *" $name "*) ;;
            *) NODE_HELPERS_LOCAL_HOSTS+=("$name") ;;
        esac
    done < <(
        {
            hostname 2>/dev/null
            hostname -s 2>/dev/null
            hostname -f 2>/dev/null
        } | awk 'NF'
    )
}

node_is_local() {
    local node="$1"
    local configured_ip
    local local_ip
    local local_host
    local ssh_host

    [[ -n "$node" ]] || return 1

    case "$node" in
        localhost|127.0.0.1)
            return 0
            ;;
    esac

    for local_host in "${NODE_HELPERS_LOCAL_HOSTS[@]}"; do
        if [[ "$node" == "$local_host" ]]; then
            return 0
        fi
    done

    if [[ -n "$NODE_HELPERS_CONFIG_FILE" ]] && command -v yq >/dev/null 2>&1; then
        configured_ip=$(yq ".host_ip.${node}" "$NODE_HELPERS_CONFIG_FILE" 2>/dev/null)
        if [[ -n "$configured_ip" && "$configured_ip" != "null" ]]; then
            for local_ip in "${NODE_HELPERS_LOCAL_IPS[@]}"; do
                if [[ "$local_ip" == "$configured_ip" ]]; then
                    return 0
                fi
            done
        fi
    fi

    if command -v ssh >/dev/null 2>&1; then
        ssh_host=$(ssh -G "$node" 2>/dev/null | awk '/^hostname / {print $2; exit}')
        if [[ -n "$ssh_host" ]]; then
            for local_host in "${NODE_HELPERS_LOCAL_HOSTS[@]}"; do
                if [[ "$ssh_host" == "$local_host" ]]; then
                    return 0
                fi
            done
            for local_ip in "${NODE_HELPERS_LOCAL_IPS[@]}"; do
                if [[ "$ssh_host" == "$local_ip" ]]; then
                    return 0
                fi
            done
        fi
    fi

    return 1
}

node_exec() {
    local node="$1"
    local command="$2"
    local local_command="$command"

    if node_is_local "$node"; then
        local_command="${local_command//${NODE_HELPERS_REMOTE_PROJECT_DIR}/${NODE_HELPERS_PROJECT_DIR}}"
        bash -lc "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && $local_command"
    else
        ssh "$node" "export PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH && $command"
    fi
}

node_check() {
    local node="$1"

    if node_is_local "$node"; then
        return 0
    fi

    ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'SSH OK'"
}

node_bash() {
    local node="$1"

    if node_is_local "$node"; then
        (cd "$NODE_HELPERS_PROJECT_DIR" && bash)
    else
        ssh "$node" bash
    fi
}

node_attach_hint() {
    local node="$1"
    local session_name="$2"

    if node_is_local "$node"; then
        echo "tmux attach -t $session_name"
    else
        echo "ssh $node 'tmux attach -t $session_name'"
    fi
}

node_list_hint() {
    local node="$1"

    if node_is_local "$node"; then
        echo "tmux ls"
    else
        echo "ssh $node 'tmux ls'"
    fi
}

node_command_hint() {
    local node="$1"
    local command="$2"
    local local_command="$command"

    if node_is_local "$node"; then
        local_command="${local_command//${NODE_HELPERS_REMOTE_PROJECT_DIR}/${NODE_HELPERS_PROJECT_DIR}}"
        echo "$local_command"
    else
        echo "ssh $node '$command'"
    fi
}