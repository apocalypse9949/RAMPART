# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for CachedEvaluator."""

from __future__ import annotations

import asyncio

import pytest

from rampart.core.evaluator import BaseEvaluator
from rampart.core.types import (
    EvalContext,
    EvalOutcome,
    EvalResult,
    Request,
    Response,
    Turn,
)
from rampart.evaluators.cache import CachedEvaluator


class MockEvaluator(BaseEvaluator):
    def __init__(self, outcome: EvalOutcome = EvalOutcome.DETECTED) -> None:
        self.call_count = 0
        self.outcome = outcome

    async def evaluate_async(self, *, context: EvalContext) -> EvalResult:
        self.call_count += 1
        # Add a small sleep to simulate work and test concurrent access
        await asyncio.sleep(0.01)
        return EvalResult(outcome=self.outcome, rationale=f"Call {self.call_count}")


@pytest.fixture
def empty_context() -> EvalContext:
    """Empty context for testing."""
    return EvalContext(turns=[])


@pytest.fixture
def context_a() -> EvalContext:
    """A context with one turn for testing."""
    return EvalContext(
        turns=[
            Turn(
                request=Request(prompt="Hello"),
                response=Response(text="Hi there!"),
            )
        ]
    )


@pytest.fixture
def context_b() -> EvalContext:
    """Another context with one turn for testing."""
    return EvalContext(
        turns=[
            Turn(
                request=Request(prompt="What time is it?"),
                response=Response(text="It is 12:00."),
            )
        ]
    )


class TestCachedEvaluator:
    async def test_caches_results(self, context_a: EvalContext) -> None:
        inner = MockEvaluator(outcome=EvalOutcome.DETECTED)
        cached = CachedEvaluator(inner=inner)

        result1 = await cached.evaluate_async(context=context_a)
        result2 = await cached.evaluate_async(context=context_a)

        assert inner.call_count == 1
        assert result1.outcome == EvalOutcome.DETECTED
        assert result2.outcome == EvalOutcome.DETECTED
        assert result1 is result2

    async def test_differentiates_contexts(
        self, context_a: EvalContext, context_b: EvalContext
    ) -> None:
        inner = MockEvaluator()
        cached = CachedEvaluator(inner=inner)

        result1 = await cached.evaluate_async(context=context_a)
        result2 = await cached.evaluate_async(context=context_b)
        result3 = await cached.evaluate_async(context=context_a)

        assert inner.call_count == 2
        assert result1 is result3
        assert result1 is not result2

    async def test_concurrent_access_only_calls_inner_once(
        self, context_a: EvalContext
    ) -> None:
        inner = MockEvaluator()
        cached = CachedEvaluator(inner=inner)

        # Fire off multiple evaluations concurrently for the same context
        tasks = [
            asyncio.create_task(cached.evaluate_async(context=context_a))
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert inner.call_count == 1
        assert all(r is results[0] for r in results)
