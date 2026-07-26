"""Live Search / Fetch / Find provider diagnostic without Discord."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os

from simajilord.domain.web import SearchDepth
from simajilord.providers.web import AiohttpPublicWebFetcher, SearxngSearchProvider
from simajilord.services.web import WebService


async def _run(args: argparse.Namespace) -> int:
    service = WebService(
        search_provider=SearxngSearchProvider(
            base_url=str(args.base_url),
            timeout_seconds=float(args.timeout),
            shared_secret=os.getenv("WEB_SEARCH_SHARED_SECRET"),
        ),
        page_fetcher=AiohttpPublicWebFetcher(timeout_seconds=float(args.timeout)),
        max_fetch_bytes=2_000_000,
    )
    try:
        ready, backend, detail = await service.status()
        options = service.search_options(
            depth=SearchDepth(str(args.depth)),
            language=str(args.language) if args.language else None,
        )
        result = await service.search(str(args.query), options)
        print(
            json.dumps(
                {
                    "status": {
                        "ready": ready,
                        "backend": backend,
                        "detail": detail,
                    },
                    "result": dataclasses.asdict(result),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if ready else 1
    finally:
        await service.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise Simajilord's local web-search provider.",
    )
    parser.add_argument("query", nargs="?", default="Python")
    parser.add_argument(
        "--base-url",
        default=os.getenv("WEB_SEARCH_BASE_URL", "http://127.0.0.1:8888"),
    )
    parser.add_argument(
        "--depth",
        choices=tuple(item.value for item in SearchDepth),
        default=SearchDepth.QUICK.value,
    )
    parser.add_argument("--language", default="")
    parser.add_argument("--timeout", type=float, default=12.0)
    raise SystemExit(asyncio.run(_run(parser.parse_args())))


if __name__ == "__main__":
    main()
