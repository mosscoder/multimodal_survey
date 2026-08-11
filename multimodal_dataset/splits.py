"""Split assignment: the test-leg rule and the tier-2 overlap computation.

The rule lives here and only here: the test set is an outermost leg of every
strip — ``leg01`` everywhere except ``site_1/strip_5``, whose western terminus
runs off both flight footprints, so its test leg is ``leg13``. Any non-test
frame whose 4 m crop intersects a test crop becomes ``train_tier_2``, carrying
the intersected share of its area in ``tier2_overlap_fraction``; everything
else is clean ``train``.
"""
from __future__ import annotations

DEFAULT_TEST_LEG = "leg01"
TEST_LEG = {("site_1", "strip_5"): "leg13"}

SPLITS = ("train", "test", "train_tier_2")


def test_leg(site: str, strip: str) -> str:
    return TEST_LEG.get((site, strip), DEFAULT_TEST_LEG)


def assign_splits(rows: list[dict]) -> None:
    """Set ``row['_split']`` and ``row['tier2_overlap_fraction']`` in place.

    Overlap is geometric and site-wide (strips can abut), computed on the
    crop polygons rows carry in ``row['_poly']`` (EPSG:6514 meters).
    """
    from shapely import unary_union
    from shapely.prepared import prep

    for row in rows:
        row["_split"] = "test" if row["leg"] == test_leg(row["site"], row["strip"]) else None
        row["tier2_overlap_fraction"] = None

    for site in {r["site"] for r in rows}:
        site_rows = [r for r in rows if r["site"] == site]
        test_union = unary_union([r["_poly"] for r in site_rows if r["_split"] == "test"])
        hits = prep(test_union)
        for row in site_rows:
            if row["_split"] == "test":
                continue
            poly = row["_poly"]
            if hits.intersects(poly):
                shared = poly.intersection(test_union).area / poly.area
                if shared > 0:
                    row["_split"] = "train_tier_2"
                    row["tier2_overlap_fraction"] = shared
                    continue
            row["_split"] = "train"
