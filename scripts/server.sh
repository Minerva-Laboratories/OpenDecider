#!/usr/bin/env bash
# OpenDecider LAN server as a systemd user service. On demand by default, like the assistant services.
#   scripts/server.sh install            install the unit, create the API key and ~/.config/opendecider/server.env
#   scripts/server.sh start|stop|restart
#   scripts/server.sh status             state, memory, address
#   scripts/server.sh logs               follow the service log
#   scripts/server.sh key                print the API key (for your clients; keep it off shared channels)
#   scripts/server.sh autostart on|off   start at login (on) or only on demand (off, default)
# Change the model / workers / port in ~/.config/opendecider/server.env, then `scripts/server.sh restart`.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
CFG=$HOME/.config/opendecider
UNIT=opendecider.service
cmd=${1:-status}; shift || true
case "$cmd" in
  install)
    mkdir -p "$CFG" "$HOME/.config/systemd/user"
    chmod 700 "$CFG"
    [ -s "$CFG/api_key" ] || { (umask 077; python3 -c "import secrets;print(secrets.token_urlsafe(32))" > "$CFG/api_key"); echo "created $CFG/api_key"; }
    [ -f "$CFG/server.env" ] || cat > "$CFG/server.env" <<EOT
# OpenDecider LAN server settings (scripts/server.sh restart after editing)
MODEL=checkpoints/opendecider-9b
BACKBONE=
MAX_BATCH=16
PORT=8000
LOG_DIR=runs/server
EOT
    sed "s|@REPO@|$REPO|" "$REPO/deploy/opendecider.service" > "$HOME/.config/systemd/user/$UNIT"
    systemctl --user daemon-reload
    echo "installed $UNIT (not started; scripts/server.sh start)" ;;
  start)
    avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    echo "MemAvailable: ${avail} GB"
    systemctl --user start $UNIT ;;
  stop|restart) systemctl --user "$cmd" $UNIT ;;
  logs) exec journalctl --user -u $UNIT -f ;;
  key) cat "$CFG/api_key" ;;
  autostart)
    case "$1" in on) systemctl --user enable $UNIT ;; off) systemctl --user disable $UNIT ;;
      *) echo "usage: $0 autostart on|off"; exit 2 ;; esac ;;
  status) ;;
  *) echo "usage: $0 install|start|stop|restart|status|logs|key|autostart on|off"; exit 2 ;;
esac
if [ "$cmd" = status ] || [ "$cmd" = start ] || [ "$cmd" = restart ]; then
  st=$(systemctl --user is-active $UNIT || true); en=$(systemctl --user is-enabled $UNIT 2>/dev/null || true)
  mem=$(systemctl --user show $UNIT -p MemoryCurrent --value 2>/dev/null)
  [[ "$mem" =~ ^[0-9]+$ ]] && mem="$((mem / 1048576)) MB" || mem="-"
  port=$(grep -E '^PORT=' "$CFG/server.env" 2>/dev/null | cut -d= -f2); port=${port:-8000}
  ip=$(hostname -I | awk '{print $1}')
  printf "%-12s %-9s autostart=%-9s mem=%s\n" opendecider "$st" "$en" "$mem"
  echo "playground: http://$ip:$port/    API: http://$ip:$port/v1/decide (Authorization: Bearer <scripts/server.sh key>)"
fi
