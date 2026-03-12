"""Tests for language parameter in filter chain."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hr_breaker.models import JobPosting, OptimizedResume, ResumeSource, FilterResult
from hr_breaker.models.language import get_language


class TestBaseFilterLanguageParam:
    """All filters accept language parameter."""

    @pytest.fixture
    def job(self):
        return JobPosting(
            title="Backend Engineer", company="Acme",
            requirements=["Python"], keywords=["python"],
        )

    @pytest.fixture
    def source(self):
        return ResumeSource(content="John Doe\nPython dev")

    @pytest.fixture
    def optimized(self, source):
        return OptimizedResume(
            html="<div>Test</div>", source_checksum=source.checksum,
            pdf_text="Test", pdf_bytes=b"pdf",
        )

    def test_content_length_checker_accepts_language(self, optimized, job, source):
        """ContentLengthChecker.evaluate accepts language param."""
        from hr_breaker.filters import ContentLengthChecker
        f = ContentLengthChecker()
        import asyncio
        asyncio.run(f.evaluate(optimized, job, source, language=get_language("ru")))

    def test_data_validator_accepts_language(self, optimized, job, source):
        """DataValidator.evaluate accepts language param."""
        from hr_breaker.filters import DataValidator
        f = DataValidator()
        import asyncio
        asyncio.run(f.evaluate(optimized, job, source, language=get_language("ru")))
