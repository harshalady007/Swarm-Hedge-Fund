"""End-to-end offline simulation over consecutive business days (about 90 by default;
set SIM_DAYS to change). sim.check_invariants asserts after every day that:
no entry on or before its signal day, no fill above the entry zone top, no stop
exit above the stop, cash never negative, at most 8 positions and 2 per sector,
at most 3 BUYs a day; and a second run on the same day changes nothing."""
import os

import sim


def test_offline_simulation(tmp_path):
    days = int(os.environ.get("SIM_DAYS", "90"))
    res = sim.simulate(str(tmp_path / "sim.db"), start="2026-03-02", days=days)
    assert res["days"] == days
    assert res["signals"].get("HOLD", 0) > 0
    if days >= 60:
        assert res["positions"] > 0, "the simulation never traded, so the fill rules were not exercised"
