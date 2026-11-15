"""Bracketed retrospective lactate reconstruction, separate from alerting."""

from math import isfinite

HOURS = (0, 6, 12, 24, 48)


def interpolate_between(first_hour, first_value, last_hour, last_value, target_hour):
    """Return a linear estimate within two observations; otherwise abstain."""
    if not first_hour < target_hour < last_hour:
        return None
    if any(v is None or not isfinite(v) or v <= 0 for v in (first_value, last_value)):
        return None
    fraction = (target_hour - first_hour) / (last_hour - first_hour)
    return first_value + fraction * (last_value - first_value)


def reconstruct(observed):
    """Map hours 6 and 12 to (value, status) using only original observed anchors."""
    recorded = {
        hour: observed[hour]
        for hour in HOURS
        if observed.get(hour) is not None
        and isfinite(observed[hour])
        and observed[hour] > 0
    }
    results = {}
    for target in (6, 12):
        if target in recorded:
            results[target] = (recorded[target], "observed")
            continue
        before = max((hour for hour in recorded if hour < target), default=None)
        after = min((hour for hour in recorded if hour > target), default=None)
        if before is None or after is None:
            results[target] = (None, "indeterminate")
            continue
        results[target] = (
            interpolate_between(
                before, recorded[before], after, recorded[after], target
            ),
            "interpolated",
        )
    return results
