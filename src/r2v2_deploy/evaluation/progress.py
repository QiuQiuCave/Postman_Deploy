"""CONTACT outcome metrics, independent of controller construction."""

CONTACT_PHASES = (
    "HOME",
    "TOP_VERTICAL",
    "TOP_RAISE",
    *(f"TOP_TURN_{i:02d}" for i in range(1, 9)),
    "TOP_HOVER",
    "TOP_DESCEND",
    "PICK_CLOSE",
    "PICK_PROBE",
    "PICK_VERIFY",
    "PICK_LIFT_OUT",
    "DELIVERY_TRANSFER",
    "DELIVERY_UPRIGHT",
    "DELIVERY_LOWER",
    "DELIVERY_LOWER_SUPPORT_PAUSE",
    "DELIVERY_RELEASE",
    "DELIVERY_RETREAT",
    "DELIVERY_SETTLE",
)


def contact_progress(report, transitions):
    """True CONTACT gates plus an explicitly mapped, non-AIR-certificate scale."""
    visited = [
        r["state"] for r in transitions if r.get("state") not in ("STAND", "FAILED")
    ]
    passed = set(visited[:-1]) if visited else set()
    current = report.get("failure_phase") or report.get("phase") or "STAND"
    completed = [p for p in CONTACT_PHASES if p in passed]
    comparable = [
        *CONTACT_PHASES[:13],
        "PICK_LIFT_OUT",
        "DELIVERY_TRANSFER",
        "DELIVERY_UPRIGHT",
        "DELIVERY_LOWER",
        "DELIVERY_RETREAT",
    ]
    aligned = 0
    for phase in comparable:
        if phase not in passed:
            break
        if phase == "PICK_LIFT_OUT" and not (
            report.get("grasp_verified") and report.get("can_exit_verified")
        ):
            break
        if phase == "DELIVERY_LOWER" and not report.get("release_commanded"):
            break
        aligned += 1
    return dict(
        schema="contact_phase_progress_v1",
        phase_order=list(CONTACT_PHASES),
        current_phase=current,
        current_phase_index=CONTACT_PHASES.index(current) + 1
        if current in CONTACT_PHASES
        else len(CONTACT_PHASES) + 1
        if current == "COMPLETE"
        else 0,
        completed_phases=len(completed),
        completed_phase_names=completed,
        air_aligned_completed_stages=aligned,
        air_comparison_scale_max=18,
        alignment_note="CONTACT measured; HOME..DESCEND=1..13; extraction=14; carry=15..18. Not an AIR result/certificate.",
    )
