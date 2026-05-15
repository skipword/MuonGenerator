from pathlib import Path

import numpy as np


def write_energy_csv(out_csv: Path, energies: np.ndarray, meta: dict):
    import csv

    E = np.asarray(energies, dtype=np.float64).reshape(-1)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# Muon CNF Simulation Results (ENERGY)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")

        writer = csv.writer(f)
        writer.writerow(["energy_GeV"])

        for e in E:
            writer.writerow([f"{float(e):.10e}"])


def write_angle_csv(out_csv: Path, theta_deg: np.ndarray, meta: dict):
    import csv

    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# Muon CNF Simulation Results (ANGLE)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")

        writer = csv.writer(f)
        writer.writerow(["theta_deg"])

        for t in th:
            s = f"{float(t):.8f}".replace(".", ",")
            writer.writerow([s])


def write_energy_shw(out_shw: Path, energies: np.ndarray, meta: dict):
    with open(out_shw, "w", encoding="utf-8") as f:
        f.write("# SHW-MVP (NOT CORSIKA/ARTI)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns: energy_GeV\n")
        for e in energies:
            f.write(f"{float(e):.10e}\n")


def write_angle_shw(out_shw: Path, theta_deg: np.ndarray, meta: dict):
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    with open(out_shw, "w", encoding="utf-8") as f:
        f.write("# SHW-MVP (NOT CORSIKA/ARTI)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns: theta_deg\n")
        for t in th:
            f.write(f"{float(t):.8f}\n")


def write_full_csv(
    out_csv: Path,
    energy_GeV: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    p_GeV_c: np.ndarray,
    px_GeV_c: np.ndarray,
    py_GeV_c: np.ndarray,
    pz_GeV_c: np.ndarray,
    meta: dict,
):
    import csv

    E = np.asarray(energy_GeV, dtype=np.float64).reshape(-1)
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    ph = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    p = np.asarray(p_GeV_c, dtype=np.float64).reshape(-1)
    px = np.asarray(px_GeV_c, dtype=np.float64).reshape(-1)
    py = np.asarray(py_GeV_c, dtype=np.float64).reshape(-1)
    pz = np.asarray(pz_GeV_c, dtype=np.float64).reshape(-1)

    n = len(E)
    if not (len(th) == len(ph) == len(p) == len(px) == len(py) == len(pz) == n):
        raise ValueError("Las columnas del CSV completo no tienen la misma longitud.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("# Muon CNF Simulation Results (FULL EVENT SAMPLE)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")

        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "energy_GeV",
            "theta_deg",
            "phi_deg",
            "p_GeV_c",
            "px_GeV_c",
            "py_GeV_c",
            "pz_GeV_c",
        ])

        for i in range(n):
            writer.writerow([
                i,
                f"{E[i]:.10e}",
                f"{th[i]:.8f}",
                f"{ph[i]:.8f}",
                f"{p[i]:.10e}",
                f"{px[i]:.10e}",
                f"{py[i]:.10e}",
                f"{pz[i]:.10e}",
            ])


def write_full_shw(
    out_shw: Path,
    energy_GeV: np.ndarray,
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    p_GeV_c: np.ndarray,
    px_GeV_c: np.ndarray,
    py_GeV_c: np.ndarray,
    pz_GeV_c: np.ndarray,
    meta: dict,
):
    E = np.asarray(energy_GeV, dtype=np.float64).reshape(-1)
    th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    ph = np.asarray(phi_deg, dtype=np.float64).reshape(-1)
    p = np.asarray(p_GeV_c, dtype=np.float64).reshape(-1)
    px = np.asarray(px_GeV_c, dtype=np.float64).reshape(-1)
    py = np.asarray(py_GeV_c, dtype=np.float64).reshape(-1)
    pz = np.asarray(pz_GeV_c, dtype=np.float64).reshape(-1)

    n = len(E)
    if not (len(th) == len(ph) == len(p) == len(px) == len(py) == len(pz) == n):
        raise ValueError("Las columnas del SHW completo no tienen la misma longitud.")

    with open(out_shw, "w", encoding="utf-8") as f:
        f.write("# SHW-MVP (NOT CORSIKA/ARTI)\n")
        for k, v in meta.items():
            f.write(f"# {k}={v}\n")
        f.write("# columns: event_id energy_GeV theta_deg phi_deg p_GeV_c px_GeV_c py_GeV_c pz_GeV_c\n")

        for i in range(n):
            f.write(
                f"{i} "
                f"{E[i]:.10e} "
                f"{th[i]:.8f} "
                f"{ph[i]:.8f} "
                f"{p[i]:.10e} "
                f"{px[i]:.10e} "
                f"{py[i]:.10e} "
                f"{pz[i]:.10e}\n"
            )


def make_zip(out_zip: Path, files: list[Path]):
    import zipfile

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)