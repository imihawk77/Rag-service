"""Small post-deployment HTTP smoke test; uses only the Python standard library."""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def main() -> int:
    base_url = os.environ.get("RAG_BASE_URL", "").rstrip("/")
    if not base_url:
        print("RAG_BASE_URL is required", file=sys.stderr)
        return 2
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        print("RAG_BASE_URL must be an absolute HTTP(S) URL", file=sys.stderr)
        return 2

    timeout = float(os.environ.get("SMOKE_TIMEOUT_SECONDS", "180"))
    deadline = time.monotonic() + timeout
    last_error = "service did not become ready"
    for attempt in range(1, 13):
        for endpoint in ("/health/live", "/health/ready"):
            try:
                with urllib.request.urlopen(base_url + endpoint, timeout=5) as response:
                    if response.status != 200:
                        raise RuntimeError(f"{endpoint}: unexpected HTTP {response.status}")
                    print(f"PASS {endpoint}: HTTP {response.status}")
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                last_error = f"{endpoint}: {exc}"
                break
        else:
            print("Deployment smoke test passed")
            return 0

        if time.monotonic() >= deadline:
            break
        print(f"Attempt {attempt}/12 failed: {last_error}; retrying")
        time.sleep(min(10, max(0, deadline - time.monotonic())))

    print(f"Deployment smoke test failed: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
