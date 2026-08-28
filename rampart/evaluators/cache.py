# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CachedEvaluator — memoizing wrapper for evaluators."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rampart.core.evaluator import BaseEvaluator

if TYPE_CHECKING:
    from rampart.core.evaluator import Evaluator
    from rampart.core.types import EvalContext, EvalResult


class CachedEvaluator(BaseEvaluator):
    """Wraps an evaluator to cache its results based on the context.

    Evaluations can be expensive, especially those backed by LLMs.
    This evaluator wraps an inner evaluator and returns cached
    results for identical contexts, preventing redundant work
    during repeated evaluation phases.

    Args:
        inner (Evaluator): The inner evaluator to cache.
    """

    def __init__(self, inner: Evaluator) -> None:
        """Initialize with an inner evaluator."""
        self._inner = inner
        self._cache: dict[int, EvalResult] = {}
        # Ensure only one underlying evaluation runs for a given context
        # at a time across concurrent calls
        self._locks: dict[int, asyncio.Lock] = {}

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        """Evaluate the context, using the cache if available.

        The cache key is generated from the string representation
        of the context's turns and manifest.

        Args:
            context (EvalContext): The evaluation context.

        Returns:
            EvalResult: The cached or newly computed evaluation result.
        """
        # We base the cache key on the string representation of turns and manifest.
        # This works well because Turn and AppManifest are dataclasses and
        # their __repr__/__str__ will capture their state deterministically.
        key = hash(str(context.turns) + str(context.manifest))

        if key in self._cache:
            return self._cache[key]

        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            # Check again in case another task completed while we waited
            if key in self._cache:
                return self._cache[key]

            result = await self._inner.evaluate_async(context=context)
            self._cache[key] = result
            return result
