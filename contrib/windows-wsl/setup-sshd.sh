#!/bin/sh
# Moves sshd inside a WSL distro to port 2222, keys only. Run it in the distro:
#
#     sudo sh setup-sshd.sh [port]
#
# Ubuntu 24.04 starts sshd from ssh.socket, which reads the port from
# sshd_config only after a daemon-reload. If the socket still doesn't listen
# on the new port, this falls back to the plain ssh.service.
set -eu

PORT="${1:-2222}"

if [ "$(id -u)" -ne 0 ]; then
    echo "run it with sudo: sudo sh $0" >&2
    exit 1
fi
if [ ! -d /run/systemd/system ]; then
    echo "systemd is not running: set systemd=true in /etc/wsl.conf, run 'wsl --shutdown' in Windows, then reopen the distro" >&2
    exit 1
fi
if ! command -v sshd >/dev/null 2>&1; then
    apt-get update
    apt-get install -y openssh-server
fi

mkdir -p /etc/ssh/sshd_config.d
# sshd keeps the first value it reads for each option, so 00- beats the
# distro's own drop-ins (50-cloud-init.conf turns passwords back on).
cat > /etc/ssh/sshd_config.d/00-orchard.conf <<EOF
Port $PORT
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
EOF

# sshd -t refuses to run without this when the socket has never started sshd.
mkdir -p /run/sshd
sshd -t

listening() {
    ss -ltn | grep -q ":$PORT "
}

systemctl daemon-reload
if systemctl is-enabled --quiet ssh.socket 2>/dev/null; then
    systemctl stop ssh.service 2>/dev/null || true
    systemctl restart ssh.socket
else
    systemctl enable --now ssh.service
    systemctl restart ssh.service
fi
sleep 1

if ! listening; then
    echo "ssh.socket did not pick up port $PORT; switching to ssh.service" >&2
    systemctl disable --now ssh.socket
    systemctl enable --now ssh.service
    systemctl restart ssh.service
    sleep 1
fi
if ! listening; then
    echo "sshd is not listening on port $PORT; check: journalctl -u ssh -n 50" >&2
    exit 1
fi
echo "sshd is listening on port $PORT"
