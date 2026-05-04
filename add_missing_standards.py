"""
add_missing_standards.py
========================
Adds IS 1331: 1971 and IS 303: 1989 to bis_all_chunks.json.
These two standards exist in the source PDF (dataset.txt) but were missed
during the initial chunk extraction, causing Q-087 and Q-091 to fail.

Usage:
    python add_missing_standards.py
"""

import json
from pathlib import Path

CHUNKS_PATH = Path("data/bis_all_chunks.json")

NEW_STANDARDS = [
    {
        "chunk_id": 565,
        "standard_id": "IS 1331: 1971",
        "title": "IS 1331 : 1971 CUT SIZES OF TIMBERS (Second Revision)",
        "category": "Timber",
        "content": (
            "IS 1331 : 1971 CUT SIZES OF TIMBERS\n"
            "(Second Revision)\n\n"
            "1. Scope — Covers specification of converted timber normally stocked in timber depot "
            "both for structural and non-structural purposes. It refers to cut sizes of timber as "
            "stocked and does not take into consideration any reduction or allowance relating to "
            "subsequent use.\n\n"
            "2. Dimensions and Tolerances\n\n"
            "2.1 Cut sizes of timber shall be grouped in terms of width and thickness or sectional "
            "area into four groups, namely, (a) batten, (b) plank, (c) scantling, and (d) baulk. "
            "The nominal sizes of width and thickness of cut sizes of timber shall be as given in Table 1.\n\n"
            "The sizes of cut timber specified in Table 1 are at a moisture content of 20 percent. "
            "A method for adjustment of dimensions at different moisture contents is given in "
            "Appendix A of the standard.\n\n"
            "2.2 Length — The preferred length of cut sizes of timber shall be 50 cm and upwards "
            "in steps of 10 cm.\n\n"
            "2.3 The measurement of length, width and thickness of cut sizes of timber shall be "
            "made on mid line of the surface on which it is measured.\n\n"
            "2.4 Tolerance — Permissible tolerances on cut sizes of timber shall be as follows:\n"
            "a) For width and thickness: Up to and including 100 mm ± 3 mm; Above 100 mm ± 6 mm\n"
            "b) For length ± 25 mm\n\n"
            "3. Grading of Cut Sizes of Timber — Cut size of timber shall be graded after seasoning "
            "at a moisture content not less than 12 percent.\n\n"
            "3.1 Grading for Structural Use — Based on permissible and prohibited defects the cut "
            "sizes of timber for structural use shall be of three grades:\n"
            "a) Grade 1 — The estimated effect in reduction of the basic strength of timber is "
            "not more than 12.5 percent.\n"
            "b) Grade 2 — The estimated effect in reduction of the basic strength of timber is "
            "not more than 25 percent.\n"
            "c) Grade 3 — The estimated effect in reduction of the basic strength of timber is "
            "not more than 37.5 percent.\n\n"
            "3.2 Grading for Non-Structural Use — Based on permissible and prohibited defects "
            "cut sizes of timber for non-structural use shall be of two grades, namely, Grade 1 "
            "and Grade 2.\n\n"
            "4. Defects\n"
            "4.1 Structural Use — Defects Prohibited: Loose grains, splits, compressive wood in "
            "coniferous timber, heart wood rot, sap rot, warp, worm holes made by power post "
            "beetles and pitch pockets shall not be permitted.\n"
            "4.2 Non-Structural Use — Defects Prohibited: Heart wood rot, sap rot, brashness, "
            "shakes, insect attack shall not be permitted.\n\n"
            "For detailed information, refer to IS 1331 : 1971 Specification for cut sizes of "
            "timbers (second revision).\n\n"
            "Keywords: timber, cut sizes, preferred dimensions, joinery, furniture, structural timber, "
            "non-structural timber, batten, plank, scantling, baulk, seasoning, grading, wood dimensions"
        ),
        "source": "SP 21 : 2005 - Summaries of Indian Standards for Building Materials"
    },
    {
        "chunk_id": 566,
        "standard_id": "IS 303: 1989",
        "title": "IS 303 : 1989 PLYWOOD FOR GENERAL PURPOSE (Third Revision)",
        "category": "Timber and Plywood",
        "content": (
            "IS 303 : 1989 PLYWOOD FOR GENERAL PURPOSE\n"
            "(Third Revision)\n\n"
            "1. Scope – Requirements of different grades and types of plywood used for general "
            "purposes.\n\n"
            "2. Grades\n"
            "a) Boiling water resistant or BWR grade, and\n"
            "b) Moisture resistant or MR grade.\n\n"
            "3. Types based on classification by appearance.\n"
            "3.1 Plywood for general purposes shall be classified into three types, namely, AA, "
            "AB and BB based on the quality of the two surfaces, namely, A and B in terms of "
            "general permissible defects.\n\n"
            "4. Materials\n"
            "4.1 Timber – Any species of timber may be used for plywood manufacture. However, "
            "a list of species for the manufacture of plywood is given in Annex B of the standard "
            "for guidance.\n"
            "4.2 Adhesive – See IS 848 : 1974. Extenders may be used with the synthetic resin "
            "adhesive (aminoresins). However, synthetic resin adhesives (aminoresins) when "
            "extended by more than 25 percent shall contain suitable preservative chemicals in "
            "sufficient concentration to satisfy the mycological test described in the standard.\n\n"
            "5. Quality — See Tables 1 and 2. Quality requirements cover defects such as blisters, "
            "checks, discoloration, dots, insect holes, joints, knots (dead and live), patches, "
            "splits, and swirl.\n\n"
            "6. Dimensions and Tolerances\n"
            "6.1 Standard dimensions: 2400x1200mm, 2100x1200mm, 1800x1200mm, 2100x900mm, "
            "1800x1200mm. Any other dimension as agreed between manufacturer and purchaser.\n"
            "6.2 Thickness: 3 ply (3,4,5,6mm), 5 ply (5,6,8,9mm), 7 ply (9,12,15,16mm), "
            "9 ply (12,15,16mm), 11 ply (19,22,25mm).\n"
            "6.3 Tolerances: Length +6/-0mm, Width +3/-0mm, Thickness <6mm ±10%, ≥6mm ±5%, "
            "Squareness 0.2%, Edge straightness 0.2%.\n\n"
            "7. Tests\n"
            "7.1 Glue adhesion — BWR grade: min average 1350N dry, 1000N mycological, 1000N water. "
            "MR grade: min average 1000N dry, 800N mycological, 800N water.\n"
            "7.2 Moisture Content — Not less than 5 percent and not more than 15 percent.\n\n"
            "For detailed information, refer to IS 303 :1989 Specification for plywood for general "
            "purposes (third revision).\n\n"
            "Keywords: plywood, general purpose, ordinary plywood, BWR grade, MR grade, boiling "
            "water resistant, moisture resistant, furniture, construction, veneer, adhesive, "
            "plywood for general purposes"
        ),
        "source": "SP 21 : 2005 - Summaries of Indian Standards for Building Materials"
    }
]


def main():
    print(f"[AddMissing] Loading {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    existing_sids = {c.get("standard_id", "") for c in chunks}

    added = 0
    for new_std in NEW_STANDARDS:
        sid = new_std["standard_id"]
        if sid in existing_sids:
            print(f"  [SKIP] {sid} already exists.")
            continue
        chunks.append(new_std)
        existing_sids.add(sid)
        added += 1
        print(f"  [ADDED] {sid}: {new_std['title'][:60]}...")

    if added > 0:
        data["chunks"] = chunks
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n[AddMissing] Done! Added {added} standards. Total: {len(chunks)}")
        print("[AddMissing] Now rebuild the index:")
        print("  python build_index.py --chunks data/bis_all_chunks.json")
    else:
        print("\n[AddMissing] No new standards to add.")


if __name__ == "__main__":
    main()
