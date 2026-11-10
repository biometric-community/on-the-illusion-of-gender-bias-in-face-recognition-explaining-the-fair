"""REPORT.md, the committed output of the full run, against the paper's findings."""

from __future__ import annotations

import re


def value(report: str, name: str) -> float:
    match = re.search(rf"- {re.escape(name)}: \*\*(-?[\d.]+)\*\*", report)
    assert match, f"{name} missing from REPORT.md"
    return float(match.group(1))


def test_recogniser_works(report, targets):
    match = re.search(r"\| lfw \| ([\d.]+)% \|", report)
    assert match
    assert float(match.group(1)) >= targets["clean_lfw_accuracy_floor"]


def test_women_have_the_higher_error_without_controls(report):
    """Sec. I: the usual finding is lower accuracy for female subjects."""
    assert value(report, "Acc MM") > value(report, "Acc FF")


def test_the_gap_vanishes_for_some_attribute_combination(report, targets):
    """Abstract: the gender gap vanishes when male and female images share certain attributes."""
    assert abs(value(report, "Filtered gap")) <= targets["vanished_gap"]
