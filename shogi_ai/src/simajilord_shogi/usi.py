"""Minimal synchronous USI loop for local integration and compatibility tests."""

from __future__ import annotations

import shlex
import sys
from dataclasses import replace
from typing import TextIO

from rsshogi.core import Board

from .branding import ENGINE_AUTHOR, ENGINE_USI_NAME
from .config import SearchConfig
from .evaluator import Evaluator
from .mcts import MCTS


class UsiEngine:
    def __init__(self, evaluator: Evaluator, search: SearchConfig) -> None:
        self.evaluator = evaluator
        self.search = search
        self.board = Board()

    def run(self, source: TextIO = sys.stdin, destination: TextIO = sys.stdout) -> None:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            response = self.handle(line)
            for output in response:
                print(output, file=destination, flush=True)
            if line == "quit":
                return

    def handle(self, line: str) -> list[str]:
        parts = shlex.split(line)
        command = parts[0]
        if command == "usi":
            return [
                f"id name {ENGINE_USI_NAME}",
                f"id author {ENGINE_AUTHOR}",
                "option name Simulations type spin "
                f"default {self.search.simulations} min 1 max 1000000000",
                "usiok",
            ]
        if command == "isready":
            return ["readyok"]
        if command == "usinewgame":
            self.board = Board()
            return []
        if command == "setoption":
            self._set_option(parts)
            return []
        if command == "position":
            self.board.set_usi_position(line[len("position ") :])
            return []
        if command == "go":
            if self.board.can_declare_win():
                return ["bestmove win"]
            if self.board.is_mated() or not self.board.legal_moves():
                return ["bestmove resign"]
            nodes = self._go_nodes(parts)
            config = replace(self.search, simulations=nodes or self.search.simulations)
            result = MCTS(self.evaluator, config).search(self.board)
            score = round(result.root_value * 1000)
            elapsed_ms = max(1, round(result.elapsed_seconds * 1000))
            return [
                f"info nodes {result.simulations} time {elapsed_ms} "
                f"nps {round(result.nodes_per_second)} score cp {score} "
                f"pv {result.best_move} string tree_peak={result.peak_tree_nodes} "
                f"tree_recycles={result.tree_recycles}",
                f"bestmove {result.best_move}",
            ]
        if command in {"stop", "ponderhit", "gameover", "quit"}:
            return []
        return [f"info string unsupported command: {command}"]

    def _set_option(self, parts: list[str]) -> None:
        if "name" not in parts or "value" not in parts:
            return
        name_index = parts.index("name") + 1
        value_index = parts.index("value") + 1
        name = " ".join(parts[name_index : value_index - 1])
        if name == "Simulations":
            self.search = replace(self.search, simulations=max(1, int(parts[value_index])))

    @staticmethod
    def _go_nodes(parts: list[str]) -> int | None:
        if "nodes" not in parts:
            return None
        index = parts.index("nodes") + 1
        return max(1, int(parts[index]))
