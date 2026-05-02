// ════════════════════════════════════════════════════════════════════════════
// Setup guide track data
// ════════════════════════════════════════════════════════════════════════════
const setupTracks = {
  mac: [
    {
      title: 'Cable + Static IPs',
      copy: 'Connect Thunderbolt 4 cables in a chain: mini1↔mini2↔mini3. On each Mac go to System Settings → Network → Thunderbolt Bridge → Details → set Configure IPv4 to Manually.',
      command: lines(
        '# Assign static IPs on each Mac:',
        '# mini1:  10.10.0.1 / 255.255.255.0',
        '# mini2:  10.10.0.2 / 255.255.255.0',
        '# mini3:  10.10.0.3 / 255.255.255.0',
        '',
        '# Verify connectivity from mini1:',
        'ifconfig | grep -A5 "bridge\\|thunderbolt"',
        'ping -c 4 10.10.0.2',
        'ping -c 4 10.10.0.3'
      ),
      diagram: 'controller'
    },
    {
      title: 'Node Inventory',
      copy: 'On mini1, copy the nodes.yaml template and fill in the alias, IP, and username for each worker node.',
      command: lines(
        'cp scripts/installations/nodes.yaml.example \\',
        '  ~/.config/smolcluster/nodes.yaml',
        '${EDITOR:-nano} ~/.config/smolcluster/nodes.yaml',
        '',
        '# nodes.yaml example:',
        '# nodes:',
        '#   - alias: mini2',
        '#     ip: 10.10.0.2',
        '#     user: your_username',
        '#   - alias: mini3',
        '#     ip: 10.10.0.3',
        '#     user: your_username'
      ),
      diagram: 'ssh'
    },
    {
      title: 'SSH Setup + Bootstrap',
      copy: 'Distribute SSH keys to all workers, install smolcluster + deps on each node, then copy your .env with W&B and HF tokens.',
      command: lines(
        '# Distribute keys and write ~/.ssh/config',
        'bash scripts/installations/setup_ssh.sh',
        '',
        '# Install deps + clone repo on each worker',
        'bash scripts/installations/setup.sh',
        '',
        '# Copy .env to workers',
        'awk \'/^[[:space:]]*-[[:space:]]*alias:/ {print $3}\' \\',
        '  ~/.config/smolcluster/nodes.yaml | while read -r node; do',
        '  scp .env "$node:~/Desktop/smolcluster/"',
        'done'
      ),
      diagram: 'keys'
    },
    {
      title: 'Configure Cluster YAMLs',
      copy: 'Update cluster_config_syncps.yaml and cluster_config_inference.yaml with your actual node aliases, IPs, and port.',
      command: lines(
        '# src/smolcluster/configs/cluster_config_syncps.yaml',
        'host_ip:',
        '  mini1: "10.10.0.1"',
        '  mini2: "10.10.0.2"',
        '  mini3: "10.10.0.3"',
        'port: 65432',
        'num_workers: 2',
        'server: mini1',
        'workers:',
        '  - hostname: mini2',
        '    rank: 1',
        '  - hostname: mini3',
        '    rank: 2'
      ),
      diagram: 'dashboard'
    },
    {
      title: 'Smoke Test',
      copy: 'Run a dry run to validate config + SSH, then launch SyncPS inference. Hit the health endpoint to confirm everything is live.',
      command: lines(
        '# Dry run (validates config + SSH, no workers launched)',
        './scripts/inference/launch_inference.sh --algorithm syncps --dry-run',
        '',
        '# Launch',
        './scripts/inference/launch_inference.sh --algorithm syncps',
        '',
        '# Health check',
        'curl http://localhost:8080/health',
        '',
        '# Generate',
        'curl -X POST http://localhost:8080/generate \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"prompt": "Once upon a time", "max_new_tokens": 50}\'',
        '',
        '# Cleanup',
        './scripts/inference/launch_inference.sh --cleanup'
      ),
      diagram: 'launch'
    }
  ],
  jetson: [
    {
      title: 'Enable SSH + Static IP',
      copy: 'On each Jetson, enable SSH and assign a static IP using nmcli. Run the discover script first to find your Ethernet interface name.',
      command: lines(
        '# Enable SSH (on each Jetson)',
        'sudo systemctl enable ssh && sudo systemctl start ssh',
        '',
        '# Find interface name',
        './scripts/installations/discover_network.sh',
        '',
        '# Assign static IP (replace CONNECTION_NAME)',
        'sudo nmcli con mod "<CONNECTION_NAME>" \\',
        '  ipv4.addresses 192.168.50.101/24 \\',
        '  ipv4.method manual',
        'sudo nmcli con up "<CONNECTION_NAME>"',
        'ip addr show'
      ),
      diagram: 'workers'
    },
    {
      title: 'Passwordless sudo',
      copy: 'setup_jetson.sh installs system packages and requires passwordless sudo. Add the NOPASSWD rule on each Jetson before running setup.sh.',
      command: lines(
        '# On each Jetson:',
        'sudo visudo',
        '',
        '# Add this line at the end (replace your_username):',
        '# your_username ALL=(ALL) NOPASSWD:ALL',
        '',
        '# Verify - should not prompt for a password:',
        'sudo whoami'
      ),
      diagram: 'deps'
    },
    {
      title: 'Node Inventory + SSH',
      copy: 'On the controller, fill nodes.yaml with Jetson IPs, then run setup_ssh.sh to distribute keys and setup.sh to bootstrap deps on every Jetson.',
      command: lines(
        'cp scripts/installations/nodes.yaml.example \\',
        '  ~/.config/smolcluster/nodes.yaml',
        '',
        '# nodes:',
        '#   - alias: jetson1',
        '#     ip: 192.168.50.101',
        '#     user: nvidia',
        '#   - alias: jetson2',
        '#     ip: 192.168.50.102',
        '#     user: nvidia',
        '',
        'bash scripts/installations/setup_ssh.sh',
        'bash scripts/installations/setup.sh'
      ),
      diagram: 'keys'
    },
    {
      title: 'Configure Cluster YAMLs',
      copy: 'Update cluster_config_syncps.yaml and cluster_config_inference.yaml with your Jetson IPs and the controller IP on the same subnet.',
      command: lines(
        '# src/smolcluster/configs/cluster_config_syncps.yaml',
        'host_ip:',
        '  mini1:   "192.168.50.100"',
        '  jetson1: "192.168.50.101"',
        '  jetson2: "192.168.50.102"',
        'port: 65432',
        'num_workers: 2',
        'server: mini1',
        'workers:',
        '  - hostname: jetson1',
        '    rank: 1',
        '  - hostname: jetson2',
        '    rank: 2'
      ),
      diagram: 'dashboard'
    },
    {
      title: 'Smoke Test',
      copy: 'Dry run first to validate config + SSH, then launch SyncPS inference across your Jetsons. Check the health endpoint.',
      command: lines(
        '# Dry run',
        './scripts/inference/launch_inference.sh --algorithm syncps --dry-run',
        '',
        '# Launch',
        './scripts/inference/launch_inference.sh --algorithm syncps',
        '',
        '# Health check',
        'curl http://localhost:8080/health',
        '',
        '# Generate',
        'curl -X POST http://localhost:8080/generate \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"prompt": "Once upon a time", "max_new_tokens": 50}\'',
        '',
        '# Cleanup',
        './scripts/inference/launch_inference.sh --cleanup'
      ),
      diagram: 'launch'
    }
  ]
};

let activeSetupTrack = 'mac';
let activeSetupStep = 0;
