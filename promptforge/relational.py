from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ContextRelation:
    """One explicit relation between two context nodes."""

    source: str
    target: str
    relation: str = "related"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("source and target must not be empty")
        if self.source == self.target:
            raise ValueError("self-relations are not supported")
        if not self.relation:
            raise ValueError("relation must not be empty")
        if not isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError("weight must be a finite non-negative number")


@dataclass(frozen=True)
class ContextRelationalProfile:
    """Descriptive graph signature for explicit context relations."""

    schema_version: str
    relation_count: int
    node_count: int
    relation_density: float
    max_degree: int
    avg_degree: float
    degree_std: float
    hub_ratio: float
    connected_components: int
    relation_kind_count: int
    relation_kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextRelationProfiler:
    """Compute a deterministic profile from caller-supplied relations.

    PromptForge does not infer semantic relations from text. Relations are an
    explicit integration input so downstream systems can distinguish measured
    structure from assumptions introduced by a model.
    """

    def profile(
        self,
        relations: Sequence[ContextRelation | Mapping[str, Any]] = (),
    ) -> ContextRelationalProfile:
        parsed = tuple(self._parse(item) for item in relations)
        unique = tuple(
            dict.fromkeys(
                (item.source, item.target, item.relation, float(item.weight))
                for item in parsed
            )
        )

        nodes = sorted({item[0] for item in unique} | {item[1] for item in unique})
        degrees = {node: 0 for node in nodes}
        adjacency: dict[str, set[str]] = {node: set() for node in nodes}
        kinds: set[str] = set()

        for source, target, relation, _weight in unique:
            degrees[source] += 1
            degrees[target] += 1
            adjacency[source].add(target)
            adjacency[target].add(source)
            kinds.add(relation)

        relation_count = len(unique)
        node_count = len(nodes)
        avg_degree = (
            sum(degrees.values()) / node_count
            if node_count else 0.0
        )
        degree_std = (
            sqrt(
                sum((degree - avg_degree) ** 2 for degree in degrees.values())
                / node_count
            )
            if node_count else 0.0
        )
        max_degree = max(degrees.values(), default=0)
        hub_ratio = max_degree / avg_degree if avg_degree > 0.0 else 0.0
        relation_density = (
            relation_count / (node_count * (node_count - 1))
            if node_count > 1 else 0.0
        )

        components = 0
        unseen = set(nodes)
        while unseen:
            components += 1
            start = min(unseen)
            unseen.remove(start)
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbour in sorted(adjacency[current]):
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        queue.append(neighbour)

        return ContextRelationalProfile(
            schema_version="context-relational.v1",
            relation_count=relation_count,
            node_count=node_count,
            relation_density=relation_density,
            max_degree=max_degree,
            avg_degree=avg_degree,
            degree_std=degree_std,
            hub_ratio=hub_ratio,
            connected_components=components,
            relation_kind_count=len(kinds),
            relation_kinds=tuple(sorted(kinds)),
        )

    @staticmethod
    def _parse(item: ContextRelation | Mapping[str, Any]) -> ContextRelation:
        if isinstance(item, ContextRelation):
            return item
        if not isinstance(item, Mapping):
            raise TypeError("relations must contain ContextRelation or mappings")
        return ContextRelation(
            source=str(item.get("source", "")),
            target=str(item.get("target", "")),
            relation=str(item.get("relation", "related")),
            weight=float(item.get("weight", 1.0)),
        )


__all__ = [
    "ContextRelation",
    "ContextRelationalProfile",
    "ContextRelationProfiler",
]
