"""Local web visualizer for the graph partitioning algorithms."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from graph_partitioning import BayesianMarkovBoundaryModel, dynamic_program_partition, greedy_partition

ROOT = Path(__file__).parent


def partition_payload(data: dict) -> dict:
    weights = [float(value) for value in data.get("weights", [])]
    parts = int(data.get("parts", 3))
    smoothness = float(data.get("smoothness", 0.35))
    model = BayesianMarkovBoundaryModel(smoothness=smoothness)
    guided = dynamic_program_partition(weights, parts, model)
    greedy = greedy_partition(weights, parts)
    probabilities = model.cut_probabilities(weights)

    def segment_rows(cuts: tuple[int, ...]) -> list[dict]:
        rows = []
        for index, (start, end) in enumerate(_segments(cuts, len(weights)), 1):
            rows.append({
                "id": index,
                "start": start,
                "end": end,
                "load": round(sum(weights[start:end]), 4),
            })
        return rows

    return {
        "weights": weights,
        "parts": parts,
        "target": round(sum(weights) / parts, 4),
        "probabilities": [round(value, 4) for value in probabilities],
        "guided": {"cuts": list(guided.cuts), "cost": round(guided.cost, 4), "segments": segment_rows(guided.cuts)},
        "greedy": {"cuts": list(greedy.cuts), "cost": round(greedy.cost, 4), "segments": segment_rows(greedy.cuts)},
    }


def _segments(cuts: tuple[int, ...], length: int) -> tuple[tuple[int, int], ...]:
    boundaries = (0, *cuts, length)
    return tuple(zip(boundaries[:-1], boundaries[1:]))


class VisualizerHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        self._send(200, "text/html; charset=utf-8", (ROOT / "visualizer.html").read_bytes())

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/partition":
            self._send(404, "application/json", b'{"error":"Not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = partition_payload(json.loads(self.rfile.read(length)))
            self._send(200, "application/json", json.dumps(payload).encode())
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send(400, "application/json", json.dumps({"error": str(error)}).encode())

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    address = ("127.0.0.1", 8765)
    print(f"Graph partition visualizer running at http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, VisualizerHandler).serve_forever()
