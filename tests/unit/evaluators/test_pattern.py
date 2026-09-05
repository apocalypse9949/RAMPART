# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the PatternEvaluator."""

from __future__ import annotations

import re

import pytest

from rampart.core.types import (
    EvalContext,
    EvalOutcome,
    Request,
    Response,
    Turn,
)
from rampart.evaluators.pattern import PatternEvaluator


class TestPatternEvaluator:
    @pytest.mark.asyncio
    async def test_detects_pattern_in_text(self) -> None:
        evaluator = PatternEvaluator(pattern=r"\b\d{3}-\d{2}-\d{4}\b", objective="SSN")
        context = EvalContext(
            turns=[
                Turn(
                    request=Request(prompt="What is the secret?"),
                    response=Response(text="My number is 123-45-6789."),
                    turn_number=0,
                ),
            ],
        )

        result = await evaluator.evaluate_async(context=context)

        assert result.outcome == EvalOutcome.DETECTED
        assert result.confidence == pytest.approx(1.0)
        assert r"\b\d{3}-\d{2}-\d{4}\b" in result.rationale
        assert result.evidence == ["123-45-6789"]

    @pytest.mark.asyncio
    async def test_not_detected_when_pattern_absent(self) -> None:
        evaluator = PatternEvaluator(pattern=r"\b\d{3}-\d{2}-\d{4}\b", objective="SSN")
        context = EvalContext(
            turns=[
                Turn(
                    request=Request(prompt="What is the secret?"),
                    response=Response(text="I cannot tell you that."),
                    turn_number=0,
                ),
            ],
        )

        result = await evaluator.evaluate_async(context=context)

        assert result.outcome == EvalOutcome.NOT_DETECTED
        assert result.confidence == pytest.approx(1.0)
        assert not result.evidence

    @pytest.mark.asyncio
    async def test_supports_compiled_regex(self) -> None:
        compiled = re.compile(r"password is (?P<pwd>\w+)", re.IGNORECASE)
        evaluator = PatternEvaluator(pattern=compiled, objective="password")
        context = EvalContext(
            turns=[
                Turn(
                    request=Request(prompt="What is the password?"),
                    response=Response(text="The Password Is Secret123"),
                    turn_number=0,
                ),
            ],
        )

        result = await evaluator.evaluate_async(context=context)

        assert result.outcome == EvalOutcome.DETECTED
        assert result.evidence == ["Password Is Secret123"]

    @pytest.mark.asyncio
    async def test_empty_transcript_returns_undetermined(self) -> None:
        evaluator = PatternEvaluator(pattern="foo", objective="foo")
        context = EvalContext(turns=[])

        result = await evaluator.evaluate_async(context=context)

        assert result.outcome == EvalOutcome.UNDETERMINED

    @pytest.mark.asyncio
    async def test_empty_response_text_returns_not_detected(self) -> None:
        evaluator = PatternEvaluator(pattern="foo", objective="foo")
        context = EvalContext(
            turns=[
                Turn(
                    request=Request(prompt="What is the secret?"),
                    response=Response(text=""),
                    turn_number=0,
                ),
            ],
        )

        result = await evaluator.evaluate_async(context=context)

        assert result.outcome == EvalOutcome.NOT_DETECTED

    def test_rejects_empty_objective(self) -> None:
        with pytest.raises(ValueError, match=r"objective.*non-empty"):
            PatternEvaluator(pattern="foo", objective="")

        with pytest.raises(ValueError, match=r"objective.*non-empty"):
            PatternEvaluator(pattern="foo", objective="   ")

    def test_rejects_empty_pattern_string(self) -> None:
        with pytest.raises(ValueError, match=r"pattern.*empty"):
            PatternEvaluator(pattern="", objective="empty")
