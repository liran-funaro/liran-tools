import json
import logging
import multiprocessing.synchronize
import os
import subprocess
import pathlib

import requests

XDG_RUNTIME_DIR = f"/run/user/{os.getuid()}"
DEFAULT_ENVS = {
    "XDG_RUNTIME_DIR": XDG_RUNTIME_DIR,
    "DBUS_SESSION_BUS_ADDRESS": f"unix:path={XDG_RUNTIME_DIR}/bus",
}
for k, v in DEFAULT_ENVS.items():
    if not os.environ.get(k):
        os.environ[k] = v

NTFY_TOPIC = os.getenv("NTFY_TOPIC")
if not NTFY_TOPIC:
    NTFY_CONF = pathlib.Path("~/.config/ntfy.topic").expanduser()
    if NTFY_CONF.exists():
        NTFY_TOPIC = NTFY_CONF.read_text("utf-8").strip()


def notify_ubuntu(app_name: str, title: str, msg: str, buttons: list[str] | tuple[str, ...] = None, level="critical"):
    cmd = ["notify-send", "-u", level, "-a", app_name]
    if buttons:
        for b in buttons:
            cmd.extend(["-A", f"{b}={b}"])
    cmd.extend([title, msg])
    p = subprocess.run(cmd, capture_output=True, encoding="utf-8")
    return p.stdout.strip()


def notify_ntfy(message: str, **kwargs):
    if not NTFY_TOPIC:
        logging.warning("No NTFY topic configured")
        return
    data = {
        "topic": NTFY_TOPIC,
        "message": message,
        **kwargs
    }
    requests.post(f"https://ntfy.sh", data=json.dumps(data).encode(encoding='utf-8'))


def listen_ntfy(stopper: multiprocessing.Event):
    with requests.get(f"https://ntfy.sh/{NTFY_TOPIC}/json", stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)
            if stopper.is_set():
                return
