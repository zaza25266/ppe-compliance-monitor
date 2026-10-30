
# src/analytics.py
# Purpose: Associate PPE detections with people and calculate
# per-person compliance statuses and dashboard summary counts.

from collections import Counter


# PPE classes our custom model was trained to detect.
PPE_CLASSES = {
    "helmet": "Helmet",
    "mask": "Mask",
    "safety vest": "Safety Vest",
    "boots": "boots",
    "glove": "glove",
}

# Initial prototype policy: require all five PPE categories.
# Change this list to match the actual workplace safety policy.
REQUIRED_PPE = list(PPE_CLASSES.keys())


def _normalize_class_name(class_name):
    """Normalize class labels so comparisons are case-insensitive."""

    # Remove surrounding whitespace and standardize capitalization.
    return str(class_name).strip().lower()


def _box_area(box):
    """Calculate the area of a bounding box."""

    # Unpack the [x1, y1, x2, y2] coordinates.
    x1, y1, x2, y2 = box

    # Prevent negative area if invalid coordinates are supplied.
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_over_person_area(ppe_box, person_box):
    """Calculate what fraction of the person's box overlaps a PPE box."""

    # Unpack the PPE and person bounding-box coordinates.
    px1, py1, px2, py2 = ppe_box
    x1, y1, x2, y2 = person_box

    # Find the intersection rectangle.
    ix1 = max(px1, x1)
    iy1 = max(py1, y1)
    ix2 = min(px2, x2)
    iy2 = min(py2, y2)

    # Compute intersection and person areas.
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    person_area = _box_area(person_box)

    # Avoid division by zero for invalid person boxes.
    if person_area == 0:
        return 0.0

    # Return the intersection relative to the person's box area.
    return intersection / person_area


def _ppe_matches_person(ppe_detection, person_detection):
    """Apply basic geometry and body-region rules to associate PPE."""

    # Read the PPE label and the two bounding boxes.
    ppe_name = _normalize_class_name(ppe_detection["class_name"])
    px1, py1, px2, py2 = ppe_detection["bbox"]
    x1, y1, x2, y2 = person_detection["bbox"]

    # Reject PPE boxes that do not overlap the person at all.
    overlap = _intersection_over_person_area(
        ppe_detection["bbox"],
        person_detection["bbox"],
    )
    if overlap <= 0:
        return False

    # Calculate the PPE box's center point.
    center_x = (px1 + px2) / 2
    center_y = (py1 + py2) / 2

    # Normalize the center within the person's bounding box.
    person_width = max(1, x2 - x1)
    person_height = max(1, y2 - y1)
    relative_x = (center_x - x1) / person_width
    relative_y = (center_y - y1) / person_height

    # Use approximate body regions to reduce obviously incorrect matches.
    # These thresholds are heuristics and may need tuning on real images.
    if ppe_name == "helmet":
        return -0.05 <= relative_x <= 1.05 and -0.05 <= relative_y <= 0.30

    if ppe_name == "mask":
        return -0.05 <= relative_x <= 1.05 and 0.05 <= relative_y <= 0.40

    if ppe_name == "safety vest":
        return -0.05 <= relative_x <= 1.05 and 0.15 <= relative_y <= 0.70

    if ppe_name == "boots":
        return -0.05 <= relative_x <= 1.05 and 0.65 <= relative_y <= 1.05

    if ppe_name == "glove":
        return -0.10 <= relative_x <= 1.10 and 0.20 <= relative_y <= 0.90

    # Unknown labels cannot be associated confidently.
    return False


def analyze_compliance(
    detections,
    required_ppe=None,
    minimum_overlap=0.0,
):
    """Return per-person PPE status and aggregate dashboard analytics."""

    # Use the configured PPE requirements, or the project defaults.
    required = [
        _normalize_class_name(item)
        for item in (required_ppe or REQUIRED_PPE)
    ]

    # Separate person and PPE detections produced by inference.py.
    people = detections.get("people", [])
    ppe_items = detections.get("ppe_items", [])

    # Prepare one result record for each detected person.
    person_results = []

    # Associate each PPE detection with the best matching person.
    for person_index, person in enumerate(people):
        assigned_ppe = set()

        for item in ppe_items:
            # Ignore labels that are not in the known PPE vocabulary.
            ppe_name = _normalize_class_name(item["class_name"])
            if ppe_name not in PPE_CLASSES:
                continue

            # Calculate overlap with this person.
            overlap = _intersection_over_person_area(
                item["bbox"],
                person["bbox"],
            )

            # Require both geometric overlap and a plausible body region.
            if (
                overlap > minimum_overlap
                and _ppe_matches_person(item, person)
            ):
                assigned_ppe.add(ppe_name)

        # Calculate which required categories were not detected for this person.
        missing_ppe = sorted(set(required) - assigned_ppe)

        # A person is compliant only when all required categories are detected.
        compliant = len(missing_ppe) == 0

        # Save the per-person result for the UI and further analysis.
        person_results.append(
            {
                "person_id": person_index + 1,
                "bbox": person["bbox"],
                "confidence": person["confidence"],
                "detected_ppe": sorted(assigned_ppe),
                "missing_ppe": missing_ppe,
                "compliant": compliant,
            }
        )

    # Count people and classify their initial prototype status.
    total_people = len(person_results)
    compliant_people = sum(p["compliant"] for p in person_results)
    non_compliant_people = total_people - compliant_people

    # Count people with each detected PPE category.
    # These are counts of people assigned an item, not raw object detections.
    people_with_ppe = {
        item: sum(item in person["detected_ppe"] for person in person_results)
        for item in PPE_CLASSES
    }

    # Count how many people are missing each required PPE category.
    people_missing_ppe = {
        item: sum(item in person["missing_ppe"] for person in person_results)
        for item in required
    }

    # Count all assigned PPE categories across people.
    assigned_ppe_counts = Counter(
        item
        for person in person_results
        for item in person["detected_ppe"]
    )

    # Calculate a percentage only when at least one person was detected.
    compliance_rate = (
        100.0 * compliant_people / total_people
        if total_people > 0
        else 0.0
    )

    # Return a consistent analytics object for the Streamlit dashboard.
    return {
        "total_people": total_people,
        "compliant_people": compliant_people,
        "non_compliant_people": non_compliant_people,
        "compliance_rate": round(compliance_rate, 2),
        "people_with_ppe": people_with_ppe,
        "people_missing_ppe": people_missing_ppe,
        "assigned_ppe_counts": dict(assigned_ppe_counts),
        "person_results": person_results,
    }
