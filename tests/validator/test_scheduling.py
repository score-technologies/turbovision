from scorevision.validator.central.scheduling import get_due_trigger_block


def test_due_trigger_requires_exact_block_on_first_observation():
    entry = {"tempo": 300, "anchor": 600}

    assert get_due_trigger_block(entry, 750) is None
    assert entry["last_observed_block"] == 750


def test_due_trigger_catches_up_a_skipped_scheduled_block():
    entry = {"tempo": 300, "anchor": 600}

    assert get_due_trigger_block(entry, 899) is None
    assert get_due_trigger_block(entry, 901) == 900
    assert get_due_trigger_block(entry, 901) is None


def test_due_trigger_does_not_repeat_after_reconnecting_past_trigger():
    entry = {"tempo": 300, "anchor": 600}

    assert get_due_trigger_block(entry, 899) is None
    assert get_due_trigger_block(entry, 900) == 900
    assert get_due_trigger_block(entry, 911) is None


def test_due_trigger_uses_newest_schedule_after_a_long_outage():
    entry = {"tempo": 300, "anchor": 600}

    assert get_due_trigger_block(entry, 899) is None
    assert get_due_trigger_block(entry, 1501) == 1500


def test_due_trigger_handles_exact_scheduled_block():
    entry = {"tempo": 300, "anchor": 600}

    assert get_due_trigger_block(entry, 600) == 600
    assert get_due_trigger_block(entry, 600) is None
