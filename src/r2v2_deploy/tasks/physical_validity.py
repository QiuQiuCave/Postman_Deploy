"""Shared physical predicates; no body controller or task-specific reward."""


def dangerous_prop_contact(penetration_m, force_N):
    return (penetration_m > 0.003) | (force_N > 150.0)


def effective_hand_control(row, weight):
    """Shared scalar/tensor truth, NOT the strict opposed/slip quality gate.

    Airborne control requires real hand contact and >=20% weight upward force.
    At the destination, contact may remain controlled as the table takes load.
    No-contact flight never qualifies, including during the recovery window.
    """
    destination = (
        row["delivery_table_contact"]
        & (row["delivery_table_xy_margin_m"] > 0.01)
        & (row["object_tilt_deg"] <= 30.0)
    )
    return (
        row["finite_state"]
        & (row["active_hand_object_contact_count"] > 0)
        & ((row["hand_vertical_force_N"] >= 0.2 * weight) | destination)
        & (row["table_contact"] == False)
        & (row["crate_object_contact"] == False)
        & (row["floor_contact"] == False)
    )


def effective_airborne_hold(row, weight):
    return effective_hand_control(row, weight) & (
        row["delivery_table_contact"] == False
    )


def stable_table_contact(row):
    """Shared scalar/tensor predicate; no measured load-fraction requirement.

    First touch alone cannot open the hand. Final success additionally requires
    the existing sustained settle interval, actual release and hand withdrawal.
    """
    return (
        row["delivery_table_contact"]
        & (row["delivery_table_xy_margin_m"] > 0.01)
        & (row["object_tilt_deg"] <= 30.0)
        & (row["object_linear_speed_mps"] < 0.02)
        & (row["object_angular_speed_radps"] < 0.15)
        & (row["crate_object_contact"] == False)
        & (row["floor_contact"] == False)
    )


def robot_stable(row, foot_limit=0.02):
    return bool(
        (foot_limit is None or row["foot_drift_m"] < foot_limit)
        and row["root_speed_mps"] < 0.03
        and row["root_angular_speed_radps"] < 0.15
        and row["base_tilt_deg"] < 35
        and row["root_height_m"] > 0.55
        and row["support_feet_count"] == 2
        and row["foot_linear_speed_mps"] < 0.025
    )
