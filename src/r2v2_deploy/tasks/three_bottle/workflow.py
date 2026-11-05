"""A continuous Loco + Cartesian IK three-bottle controller."""

from . import (
    geometry,
    initialization,
    arm_control,
    crate_control,
    grasp,
    handoff,
    measurements,
    placement,
    return_path,
    sequence,
)


class Workflow:
    """Own the physical world and advance it by one 1 ms integration step."""

    seeded_ik_step = staticmethod(geometry.bounded_step)
    regrasp_ik_step = staticmethod(geometry.bounded_step)
    bottle_clearance = staticmethod(geometry.bottle_bottom_clearance)
    begin_source_stance = sequence.sequence_begin_source_stance
    configure_repeat_handoff = sequence.sequence_configure_repeat_handoff
    prepare_source_cartesian = sequence.sequence_prepare_source_cartesian
    rotate_goal = sequence.sequence_rotate_goal
    station_object_frame = sequence.sequence_station_object_frame
    start_shelf_regrip = sequence.sequence_start_shelf_regrip
    begin_natural_station_trim = sequence.sequence_begin_natural_station_trim
    natural_station_targets = sequence.sequence_natural_station_targets
    natural_rest_goals = sequence.sequence_natural_rest_goals
    natural_posture_joints = sequence.sequence_natural_posture_joints
    advance = sequence.sequence_advance
    loaded_targets = sequence.sequence_loaded_targets
    dual_carry_goals = sequence.sequence_dual_carry_goals
    dual_targets = sequence.sequence_dual_targets
    start_return = sequence.sequence_start_return
    empty_hand_crate_rim_clearance = sequence.sequence_empty_hand_crate_rim_clearance
    empty_targets = sequence.sequence_empty_targets
    plan = arm_control.sequence_plan
    safety = measurements.sequence_safety
    step = sequence.sequence_step
    begin_open_back = return_path.departure_begin_open_back
    begin_safe_place = placement.right_release_begin_safe_place
    safe_third_station = placement.workspace_safe_third_station
    switch_active = placement.workspace_switch_active
    start_next_pick = grasp.grasp_stance_start_next_pick
    trim_targets = grasp.adaptive_grasp_trim_targets
    start_practical_place = placement.shelf_heading_start_practical_place
    load_place_initialization = grasp.initial_layout_load_place_initialization
    adjust_station_path = placement.station_depth_adjust_station_path
    begin_walk = placement.third_place_begin_walk
    start_third_functional = placement.third_place_start_third_functional
    station_grasp_targets = placement.staging_station_grasp_targets
    torso_pose = placement.front_shelf_torso_pose
    begin_place = placement.third_place_clearance_begin_place
    start_left_after_stance = placement.left_stance_start_left_after_stance
    start_stance_arc = return_path.return_heading_start_stance_arc
    enable_carry = handoff.carry_enable_carry
    both_ready = handoff.carry_both_ready
    begin_dual = handoff.carry_begin_dual
    upright_place_targets = handoff.carry_upright_place_targets
    shelf_withdraw_targets = handoff.carry_shelf_withdraw_targets
    configure_upright_carry = sequence.repeat_configure_upright_carry
    start_post = sequence.repeat_start_post
    start_second = sequence.repeat_start_second
    extra_goals = sequence.repeat_extra_goals
    refine_targets = sequence.repeat_refine_targets
    all_bottles = measurements.inventory_all_bottles
    inventory_guard = measurements.inventory_inventory_guard
    begin_route = placement.rack_begin_route
    right_rest_goal = placement.rack_right_rest_goal
    progress_back = placement.shelf_progress_back
    dual_measure = measurements.shelf_dual_measure
    shelf_support = measurements.shelf_shelf_support
    shelf_stable = measurements.shelf_shelf_stable
    native_measure = measurements.loaded_route_native_measure
    rack_measure = measurements.loaded_route_rack_measure
    begin_brake = placement.loaded_route_begin_brake
    goal_command = placement.loaded_route_goal_command
    ready_move = handoff.regrasp_ready_move
    object_pose = handoff.handoff_object_pose
    errors = handoff.handoff_errors
    set_frame = grasp.grasp_gate_set_frame
    activate_native = grasp.grasp_gate_activate_native
    log_stance = grasp.stance_log_stance
    make_measurement_proxy = measurements.control_make_measurement_proxy
    prepare_empty_hands = sequence.control_prepare_empty_hands
    home_at = sequence.control_home_at
    measure = measurements.control_measure
    diagnostic_hold = crate_control.crate_route_diagnostic_hold
    placement_good = crate_control.crate_transport_placement_good
    loaded_guard = crate_control.crate_support_loaded_guard
    retained = crate_control.crate_retention_retained
    switch = crate_control.crate_delivery_switch
    bottle_metrics = measurements.crate_delivery_bottle_metrics
    physically_supported = measurements.crate_delivery_physically_supported
    pre_release_good = crate_control.crate_delivery_pre_release_good
    base_yaw = crate_control.loaded_motion_base_yaw
    event = crate_control.loaded_motion_event
    enter = crate_control.crate_motion_enter
    stable = crate_control.crate_motion_stable
    held_goals = crate_control.crate_motion_held_goals
    commanded_goals = crate_control.crate_motion_commanded_goals
    capture_carry = crate_control.crate_motion_capture_carry
    carry_targets = crate_control.crate_motion_carry_targets
    configure_inverse_frame = crate_control.crate_motion_configure_inverse_frame
    inverse_targets = crate_control.crate_motion_inverse_targets
    sync = crate_control.crate_sync
    __init__ = initialization.initialize
