import time
from datetime import datetime

import psutil

PROCESS_MATCH = ["uvicorn", "app.main:app"]
INTERVAL_SECONDS = 1.0


def bytes_to_mb(value: float) -> float:
    return value / (1024 * 1024)


def find_backend_process():
    candidates = []

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(cmdline).lower()

            if all(token.lower() in joined for token in PROCESS_MATCH):
                candidates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not candidates:
        return None

    return candidates[0]


def collect_process_tree(proc: psutil.Process):
    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return procs


def main():
    proc = find_backend_process()
    if proc is None:
        print("No encontré el proceso del backend. Arráncalo primero con:")
        print("python -m uvicorn app.main:app")
        return

    print(f"Monitoreando PID={proc.pid}")
    print("Presiona Ctrl+C para detener.\n")

    peak_rss = 0.0
    peak_cpu = 0.0

    for p in collect_process_tree(proc):
        try:
            p.cpu_percent(interval=None)
        except Exception:
            pass

    try:
        while True:
            tree = collect_process_tree(proc)

            total_rss = 0
            total_cpu = 0.0

            for p in tree:
                try:
                    mem = p.memory_info().rss
                    cpu = p.cpu_percent(interval=None)
                    total_rss += mem
                    total_cpu += cpu
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            rss_mb = bytes_to_mb(total_rss)
            peak_rss = max(peak_rss, rss_mb)
            peak_cpu = max(peak_cpu, total_cpu)

            now = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{now}] RAM actual: {rss_mb:8.2f} MB | "
                f"RAM pico: {peak_rss:8.2f} MB | "
                f"CPU actual: {total_cpu:6.2f}% | "
                f"CPU pico: {peak_cpu:6.2f}%"
            )

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nMonitoreo detenido.")
        print(f"RAM pico observada: {peak_rss:.2f} MB")
        print(f"CPU pico observada: {peak_cpu:.2f}%")


if __name__ == "__main__":
    main()