import asyncio
import time
import statistics
from typing import Any

import httpx

URL = "http://127.0.0.1:8000/simulate-full"

# Reemplaza estos valores por unos que ya sepas que funcionan en tu app
PAYLOAD = {
    "bx": 26.5,
    "bz": 35.2,
    "altura": 5200.0
}

# Configuración de la prueba
TOTAL_REQUESTS = 5     # total de solicitudes a lanzar
CONCURRENT_REQUESTS = 1  # cuántas salen al mismo tiempo
TIMEOUT_SECONDS = 300.0     # importante: tus simulaciones pueden tardar bastante


async def make_request(
    client: httpx.AsyncClient,
    request_id: int,
    sem: asyncio.Semaphore
) -> dict[str, Any]:
    async with sem:
        start = time.perf_counter()
        try:
            response = await client.post(URL, json=PAYLOAD)
            elapsed = time.perf_counter() - start

            result: dict[str, Any] = {
                "request_id": request_id,
                "ok": response.status_code == 200,
                "status_code": response.status_code,
                "elapsed_s": elapsed,
            }

            try:
                data = response.json()
                result["message"] = data.get("message", "")
                result["simulation_time_s"] = data.get("simulation_time_s", None)
                result["run_id"] = data.get("run_id", "")
            except Exception:
                result["message"] = response.text[:300]

            return result

        except Exception as e:
            elapsed = time.perf_counter() - start
            return {
                "request_id": request_id,
                "ok": False,
                "status_code": None,
                "elapsed_s": elapsed,
                "message": repr(e),
            }


async def main():
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    timeout = httpx.Timeout(TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            make_request(client, i + 1, sem)
            for i in range(TOTAL_REQUESTS)
        ]

        overall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        overall_elapsed = time.perf_counter() - overall_start

    ok_results = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]
    elapsed_values = [r["elapsed_s"] for r in results]

    print("\n" + "=" * 60)
    print("RESULTADOS INDIVIDUALES")
    print("=" * 60)

    for r in results:
        print(
            f"Req #{r['request_id']:02d} | "
            f"ok={r['ok']} | "
            f"status={r['status_code']} | "
            f"t={r['elapsed_s']:.2f}s | "
            f"msg={r.get('message', '')}"
        )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"URL: {URL}")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"Exitosas: {len(ok_results)}")
    print(f"Fallidas: {len(fail_results)}")
    print(f"Tiempo total de la prueba: {overall_elapsed:.2f}s")

    if elapsed_values:
        print(f"Tiempo mínimo: {min(elapsed_values):.2f}s")
        print(f"Tiempo máximo: {max(elapsed_values):.2f}s")
        print(f"Tiempo promedio: {statistics.mean(elapsed_values):.2f}s")
        if len(elapsed_values) > 1:
            print(f"Mediana: {statistics.median(elapsed_values):.2f}s")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())