"""
StarkDB Client
Zero-dependency Python client connecting to StarkDB via local TCP socket (127.0.0.1:9099)
or Unix Domain Socket (/run/starkos/starkdb.sock on Linux).
Version 1.0.0 (Ultimate Legacy)
"""

import json
import socket
from typing import Any, Dict, List, Optional


class StarkDBSession:
    """Persistent socket session for high-throughput batch operations."""
    def __init__(self, host: str = "127.0.0.1", port: int = 9099, socket_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None
        self._file = None

    def __enter__(self):
        if self.socket_path:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self.socket_path)
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect((self.host, self.port))
        self._file = self._sock.makefile("r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        if self._sock:
            self._sock.close()

    def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        line = json.dumps(payload) + "\n"
        self._sock.sendall(line.encode("utf-8"))
        resp_line = self._file.readline()
        if not resp_line:
            raise ConnectionError("Connection closed by StarkDB server")
        return json.loads(resp_line)


class StarkDBClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9099, socket_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.socket_path = socket_path

    def session(self) -> StarkDBSession:
        return StarkDBSession(self.host, self.port, self.socket_path)

    def _send_cmd(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON command line and return parsed JSON response."""
        with self.session() as sess:
            return sess.send(payload)

    def ping(self) -> str:
        res = self._send_cmd({"cmd": "ping"})
        return res.get("message", "")

    def list_databases(self) -> List[Dict[str, Any]]:
        res = self._send_cmd({"cmd": "list_databases"})
        return res.get("databases", [])

    def use_database(self, name: str) -> Dict[str, Any]:
        return self._send_cmd({"cmd": "use_database", "name": name})

    def create_database(self, name: str, dims: Optional[int] = None) -> Dict[str, Any]:
        payload = {"cmd": "create_database", "name": name}
        if dims is not None:
            payload["dims"] = dims
        return self._send_cmd(payload)

    def upsert_node(
        self,
        name: str,
        node_type: Optional[str] = None,
        tags: Optional[str] = None,
        content: Optional[str] = None,
        base_weight: Optional[float] = None,
        pinned: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload = {"cmd": "upsert_node", "name": name}
        if node_type is not None:
            payload["node_type"] = node_type
        if tags is not None:
            payload["tags"] = tags
        if content is not None:
            payload["content"] = content
        if base_weight is not None:
            payload["base_weight"] = base_weight
        if pinned is not None:
            payload["pinned"] = pinned
        return self._send_cmd(payload)

    def remember(self, source: str, relation: str, target: str, weight: float = 0.9) -> int:
        res = self._send_cmd({
            "cmd": "remember",
            "source": source,
            "relation": relation,
            "target": target,
            "weight": weight,
        })
        if res.get("status") == "ok":
            return res.get("synapse_id", 0)
        raise RuntimeError(res.get("message", "Error in remember"))

    def activate(self, concept: str, depth: int = 2, decay: float = 0.7, threshold: float = 0.1) -> List[Dict[str, Any]]:
        res = self._send_cmd({
            "cmd": "activate",
            "concept": concept,
            "depth": depth,
            "decay": decay,
            "threshold": threshold,
        })
        return res.get("activated", [])

    def attach_vector(self, node: str, vector: List[float]) -> int:
        res = self._send_cmd({
            "cmd": "attach_vector",
            "node": node,
            "vector": vector,
        })
        return res.get("node_id", 0)

    def recall(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        res = self._send_cmd({
            "cmd": "recall",
            "vector": vector,
            "top_k": top_k,
        })
        return res.get("vector_matches", [])

    def hybrid(self, vector: List[float], top_vectors: int = 3, depth: int = 2) -> List[Dict[str, Any]]:
        res = self._send_cmd({
            "cmd": "hybrid",
            "vector": vector,
            "top_vectors": top_vectors,
            "depth": depth,
        })
        return res.get("activated", [])

    def bound_attack(
        self,
        text: Optional[str] = None,
        vector: Optional[List[float]] = None,
        anchors: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        node_type: Optional[str] = None,
        top_k: int = 5,
        depth: int = 2,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"cmd": "bound_attack", "top_k": top_k, "depth": depth}
        if text: payload["text"] = text
        if vector: payload["vector"] = vector
        if anchors: payload["anchors"] = anchors
        if tags: payload["tags"] = tags
        if node_type: payload["node_type"] = node_type
        if weights: payload["weights"] = weights
        res = self._send_cmd(payload)
        return res.get("bound_hits", [])

    def text_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        res = self._send_cmd({"cmd": "text_search", "query": query, "top_k": top_k})
        return res.get("text_matches", [])

    def find_nodes(
        self,
        search: Optional[str] = None,
        node_type: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        payload = {"cmd": "find_nodes", "limit": limit}
        if search: payload["search"] = search
        if node_type: payload["node_type"] = node_type
        if tag: payload["tag"] = tag
        res = self._send_cmd(payload)
        return res.get("nodes", [])

    def stats(self) -> Dict[str, Any]:
        res = self._send_cmd({"cmd": "stats"})
        return res.get("stats", {})

    def compact(self, name: Optional[str] = None) -> Dict[str, Any]:
        payload = {"cmd": "compact"}
        if name: payload["name"] = name
        return self._send_cmd(payload)

    def consolidate(self, half_life_days: float = 30.0, prune_below: float = 0.05) -> int:
        res = self._send_cmd({
            "cmd": "consolidate",
            "half_life_days": half_life_days,
            "prune_below": prune_below,
        })
        return res.get("pruned", 0)


if __name__ == "__main__":
    client = StarkDBClient()
    print("Connecting to StarkDB...")
    try:
        print("Ping:", client.ping())
        print("Stats:", client.stats())
    except ConnectionRefusedError:
        print("StarkDB daemon is not running. Start it with: cargo run -- serve")
