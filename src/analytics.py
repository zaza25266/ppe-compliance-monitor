# Purpose: Associate detected PPE items with people using approximate
# bounding-box geometry and calculate potential compliance statistics.

PPE_CLASSES = {
    "helmet": "Helmet",
    "mask": "Mask",
    "safety vest": "Safety Vest",
    "boots": "boots",
    "glove": "glove"
}

REQUIRED_PPE = list(PPE_CLASSES.keys())


def normalize_name(name):
    # Normalize class names for consistent comparison.
    return str(name).strip().lower()


def ppe_matches_person(person, item):
    # Estimate whether a PPE detection belongs to a person's body region.
    px1, py1, px2, py2 = person["bbox"]
    x1, y1, x2, y2 = item["bbox"]

    pw = max(px2 - px1, 1)
    ph = max(py2 - py1, 1)

    # Calculate PPE center relative to the person bounding box.
    cx = ((x1 + x2) / 2 - px1) / pw
    cy = ((y1 + y2) / 2 - py1) / ph

    name = normalize_name(item["class_name"])

    # Approximate vertical body regions.
    regions = {
        "helmet": (0.0, 0.30),
        "mask": (0.05, 0.40),
        "safety vest": (0.15, 0.75),
        "boots": (0.65, 1.10),
        "glove": (0.20, 0.90)
    }

    if name not in regions:
        return False

    low, high = regions[name]

    # Require the PPE center to be inside the person's horizontal area.
    return -0.10 <= cx <= 1.10 and low <= cy <= high


def analyze_compliance(detections, required_ppe=None):
    # Associate PPE detections with each person and calculate statistics.
    requirements = REQUIRED_PPE if required_ppe is None else required_ppe
    required = [normalize_name(item) for item in requirements]

    people = detections["people"]
    ppe_items = detections["ppe_items"]

    person_results = []
    assigned_counts = {name: 0 for name in PPE_CLASSES}

    for index, person in enumerate(people):
        assigned = set()

        for item in ppe_items:
            name = normalize_name(item["class_name"])

            if name in PPE_CLASSES and ppe_matches_person(person, item):
                assigned.add(name)

        for name in assigned:
            assigned_counts[name] += 1

        missing = [name for name in required if name not in assigned]

        person_results.append({
            "person_id": index + 1,
            "assigned_ppe": sorted(assigned),
            "missing_ppe": missing,
            "potentially_compliant": len(missing) == 0
        })

    total = len(people)
    compliant = sum(
        result["potentially_compliant"] for result in person_results
    )

    return {
        "total_people": total,
        "compliant_people": compliant,
        "non_compliant_people": total - compliant,
        "compliance_rate": compliant / total * 100 if total else 0,
        "assigned_ppe_counts": assigned_counts,
        "person_results": person_results
    }
