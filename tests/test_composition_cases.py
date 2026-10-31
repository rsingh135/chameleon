"""Authored campaign fixtures and independent diagnostics must be trustworthy."""

from copy import deepcopy

import pytest

from tools.composition_cases import authored_diagnostics, generate_cases, validate_source


@pytest.mark.parametrize("index", [0, 1])
def test_authored_sources_are_valid_and_have_disjoint_campaigns(index):
    task = generate_cases()[index]
    validate_source(task)
    campaigns = task["expected_campaigns"]
    memberships = [set(campaign["element_ids"]) for campaign in campaigns]
    assert len(campaigns) == 2
    assert memberships[0].isdisjoint(memberships[1])
    assert len(memberships[0] | memberships[1]) == len(task["source"]["elements"]) - 1
    assert all(len(campaign["badge_groups"]) == 2 for campaign in campaigns)
    assert all(element["id"].startswith("n") for element in task["source"]["elements"])
    diagnosis = authored_diagnostics(task, task["source"])
    assert diagnosis["reading_order_failures"] == 0
    assert diagnosis["detached_pairs"] == 0
    assert diagnosis["ambiguous_pairs"] == 0


def test_fixture_generation_is_deterministic_and_returns_independent_values():
    first, second = generate_cases(), generate_cases()
    assert first == second
    first[0]["source"]["elements"][1]["x"] = 999
    first[0]["targets"][0]["width"] = 7
    assert second == generate_cases()
    assert first[1] == second[1]


def test_authored_ownership_detects_an_action_moved_to_the_other_campaign():
    task = generate_cases()[0]
    design = deepcopy(task["source"])
    moved_ids = set(task["expected_campaigns"][0]["action_ids"])
    for element in design["elements"]:
        if element["id"] in moved_ids:
            element["x"] += 800
    result = authored_diagnostics(task, design)
    assert result["detached_pairs"] == 1
    assert not result["campaigns"][0]["own_action_message_pair_present"]


def test_authored_order_detects_body_before_its_heading_without_using_inferred_graph():
    task = generate_cases()[0]
    design = deepcopy(task["source"])
    body_ids = set(task["expected_campaigns"][0]["body_ids"])
    for element in design["elements"]:
        if element["id"] in body_ids:
            element["y"] = 500
    result = authored_diagnostics(task, design)
    assert result["reading_order_failures"] == 1
    assert result["campaigns"][0]["reading_order_failures"] == ["headline_before_body"]
