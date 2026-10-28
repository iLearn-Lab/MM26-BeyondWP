from turtle import heading
from typing import Any, List, Optional, Tuple
import math
import numpy as np
import random
from gym import spaces
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.core.simulator import Observations
from habitat.tasks.nav.nav import TeleportAction
from habitat_extensions.utils import compute_heading_to, rtheta_to_global_coordinates


@registry.register_task_action
class GoTowardPoint(TeleportAction):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rotate_agent = self._config.rotate_agent

    def step(self, *args: Any, r: float, theta: float, **kwargs: Any) -> Observations:
        y_delta = kwargs["y_delta"] if "y_delta" in kwargs else 0.0
        pos = rtheta_to_global_coordinates(self._sim, r, theta, y_delta=y_delta, dimensionality=3)
        agent_pos = self._sim.get_agent_state().position
        new_pos = np.array(self._sim.step_filter(agent_pos, pos))
        new_rot = self._sim.get_agent_state().rotation
        if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
            new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)
        else:
            new_pos = np.array(self._sim.pathfinder.snap_point(new_pos))
            if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
                new_pos = agent_pos
            if self._rotate_agent:
                new_rot, _ = compute_heading_to(agent_pos, pos)
        assert np.all(np.isfinite(new_pos))
        return self._sim.get_observations_at(
            position=new_pos, rotation=new_rot, keep_agent_at_new_pose=True
        )

    @property
    def action_space(self) -> spaces.Dict:
        coord_range = self.COORDINATE_MAX - self.COORDINATE_MIN
        return spaces.Dict(
            {
                "r": spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([np.sqrt(2 * coord_range**2)]),
                    dtype=np.float,
                ),
                "theta": spaces.Box(
                    low=np.array([0.0]), high=np.array([2 * np.pi]), dtype=np.float
                ),
            }
        )


@registry.register_task_action
class MoveHighToLowAction(SimulatorTaskAction):

    def cal_heading(agent_state):
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return math.degrees(heading)

    def turn(self, angle):
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount
        angle = round(angle / turn_unit) * turn_unit
        if 180 < angle <= 360:
            angle -= 360
        if angle >= 0:
            turn_actions = [left_action] * (angle // turn_unit)
        else:
            turn_actions = [right_action] * (-angle // turn_unit)
        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)

    def step(self, *args: Any, angle: float, distance: float, niu1niu: bool = False, **kwargs: Any):
        if not niu1niu:
            init_state = self._sim.get_agent_state()
            forward_action = HabitatSimActions.MOVE_FORWARD
            init_forward = (
                self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount
            )
            theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            self._sim.set_agent_state(init_state.position, rotation)
            ksteps = int(distance // init_forward)
            for k in range(ksteps):
                if k == ksteps - 1:
                    output = self._sim.step(forward_action)
                else:
                    self._sim.step_without_obs(forward_action)
            return output
        elif niu1niu:
            positions = []
            collisions = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            left_action = HabitatSimActions.TURN_LEFT
            right_action = HabitatSimActions.TURN_RIGHT
            foward_unit = (
                self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount
            )
            angle = math.degrees(angle)
            self.turn(angle)
            forward_step = int(distance // foward_unit)
            for k in range(forward_step):
                if k == forward_step - 1:
                    output = self._sim.step(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    break
                else:
                    self._sim.step_without_obs(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    if self._sim.previous_step_collided:
                        output = self._sim.step(forward_action)
                        break
            forward_step = forward_step - len(collisions)
            if forward_step > 0:
                init_try_angle = random.choice([90, 270])
                self.turn(init_try_angle)
                if init_try_angle == 90:
                    turn_seqs = [
                        {"head_turns": [], "tail_turns": [right_action] * 3},
                        {"head_turns": [right_action], "tail_turns": [right_action] * 2},
                        {"head_turns": [right_action], "tail_turns": [right_action]},
                        {"head_turns": [right_action] * 2, "tail_turns": [left_action]},
                        {"head_turns": [right_action], "tail_turns": [left_action] * 2},
                        {"head_turns": [right_action], "tail_turns": [left_action] * 3},
                    ]
                elif init_try_angle == 270:
                    turn_seqs = [
                        {"head_turns": [], "tail_turns": [left_action] * 3},
                        {"head_turns": [left_action], "tail_turns": [left_action] * 2},
                        {"head_turns": [left_action], "tail_turns": [left_action]},
                        {"head_turns": [left_action] * 2, "tail_turns": [right_action]},
                        {"head_turns": [left_action], "tail_turns": [right_action] * 2},
                        {"head_turns": [left_action], "tail_turns": [right_action] * 3},
                    ]
                for turn_seq in turn_seqs:
                    for turn in turn_seq["head_turns"]:
                        self._sim.step_without_obs(turn)
                    prev_position = self._sim.get_agent_state().position
                    self._sim.step_without_obs(forward_action)
                    post_posiiton = self._sim.get_agent_state().position
                    if list(prev_position) != list(post_posiiton):
                        positions.append(self._sim.get_agent_state().position)
                        collisions.append(self._sim.previous_step_collided)
                        for turn in turn_seq["tail_turns"]:
                            self._sim.step_without_obs(turn)
                        for k in range(forward_step):
                            if k == forward_step - 1:
                                output = self._sim.step(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                            else:
                                self._sim.step_without_obs(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                                if self._sim.previous_step_collided:
                                    output = self._sim.step(forward_action)
                                    break
                        break
            return output


@registry.register_task_action
class MoveHighToLowActionEval(SimulatorTaskAction):

    @staticmethod
    def cal_heading(agent_state):
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return math.degrees(heading)

    def turn(self, angle):
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount
        angle = round(angle / turn_unit) * turn_unit
        if 180 < angle <= 360:
            angle -= 360
        if angle >= 0:
            turn_actions = [left_action] * (angle // turn_unit)
        else:
            turn_actions = [right_action] * (-angle // turn_unit)
        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)

    def step(self, *args: Any, angle: float, distance: float, niu1niu: bool = False, **kwargs: Any):
        if not niu1niu:
            init_state = self._sim.get_agent_state()
            positions = []
            collisions = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            init_forward = (
                self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount
            )
            theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            self._sim.set_agent_state(init_state.position, rotation)
            ksteps = int(distance // init_forward)
            for k in range(ksteps):
                if k == ksteps - 1:
                    output = self._sim.step(forward_action)
                else:
                    self._sim.step_without_obs(forward_action)
                positions.append(self._sim.get_agent_state().position)
                collisions.append(self._sim.previous_step_collided)
            output["positions"] = positions
            output["collisions"] = collisions
            return output
        elif niu1niu:
            positions = []
            collisions = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            left_action = HabitatSimActions.TURN_LEFT
            right_action = HabitatSimActions.TURN_RIGHT
            foward_unit = (
                self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount
            )
            angle = math.degrees(angle)
            self.turn(angle)
            forward_step = int(distance // foward_unit)
            for k in range(forward_step):
                if k == forward_step - 1:
                    output = self._sim.step(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    break
                else:
                    self._sim.step_without_obs(forward_action)
                    positions.append(self._sim.get_agent_state().position)
                    collisions.append(self._sim.previous_step_collided)
                    if self._sim.previous_step_collided:
                        output = self._sim.step(forward_action)
                        break
            forward_step = forward_step - len(collisions)
            if forward_step > 0:
                init_try_angle = random.choice([90, 270])
                self.turn(init_try_angle)
                if init_try_angle == 90:
                    turn_seqs = [
                        {"head_turns": [], "tail_turns": [right_action] * 3},
                        {"head_turns": [right_action], "tail_turns": [right_action] * 2},
                        {"head_turns": [right_action], "tail_turns": [right_action]},
                        {"head_turns": [right_action] * 2, "tail_turns": [left_action]},
                        {"head_turns": [right_action], "tail_turns": [left_action] * 2},
                        {"head_turns": [right_action], "tail_turns": [left_action] * 3},
                    ]
                elif init_try_angle == 270:
                    turn_seqs = [
                        {"head_turns": [], "tail_turns": [left_action] * 3},
                        {"head_turns": [left_action], "tail_turns": [left_action] * 2},
                        {"head_turns": [left_action], "tail_turns": [left_action]},
                        {"head_turns": [left_action] * 2, "tail_turns": [right_action]},
                        {"head_turns": [left_action], "tail_turns": [right_action] * 2},
                        {"head_turns": [left_action], "tail_turns": [right_action] * 3},
                    ]
                for turn_seq in turn_seqs:
                    for turn in turn_seq["head_turns"]:
                        self._sim.step_without_obs(turn)
                    prev_position = self._sim.get_agent_state().position
                    self._sim.step_without_obs(forward_action)
                    post_posiiton = self._sim.get_agent_state().position
                    if list(prev_position) != list(post_posiiton):
                        positions.append(self._sim.get_agent_state().position)
                        collisions.append(self._sim.previous_step_collided)
                        for turn in turn_seq["tail_turns"]:
                            self._sim.step_without_obs(turn)
                        for k in range(forward_step):
                            if k == forward_step - 1:
                                output = self._sim.step(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                            else:
                                self._sim.step_without_obs(forward_action)
                                positions.append(self._sim.get_agent_state().position)
                                collisions.append(self._sim.previous_step_collided)
                                if self._sim.previous_step_collided:
                                    output = self._sim.step(forward_action)
                                    break
                        break
            output["positions"] = positions
            output["collisions"] = collisions
            return output


@registry.register_task_action
class MoveHighToLowActionInference(SimulatorTaskAction):

    def turn(self, angle):
        left_action = HabitatSimActions.TURN_LEFT
        right_action = HabitatSimActions.TURN_RIGHT
        turn_unit = self._sim.get_agent(0).agent_config.action_space[left_action].actuation.amount
        angle = round(angle / turn_unit) * turn_unit
        if 180 < angle <= 360:
            angle -= 360
        if angle >= 0:
            turn_actions = [left_action] * (angle // turn_unit)
        else:
            turn_actions = [right_action] * (-angle // turn_unit)
        for turn_action in turn_actions:
            self._sim.step_without_obs(turn_action)

    def get_agent_info(self):
        agent_state = self._sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {"position": agent_state.position.tolist(), "heading": heading, "stop": False}

    def step(self, *args: Any, angle: float, distance: float, niu1niu: bool = False, **kwargs: Any):
        if not niu1niu:
            init_state = self._sim.get_agent_state()
            cur_path = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            init_forward = (
                self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount
            )
            theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
            rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
            self._sim.set_agent_state(init_state.position, rotation)
            ksteps = int(distance // init_forward)
            for k in range(ksteps):
                if k == ksteps - 1:
                    output = self._sim.step(forward_action)
                else:
                    self._sim.step_without_obs(forward_action)
                cur_path.append(self.get_agent_info())
            output["cur_path"] = cur_path
            return output
        elif niu1niu:
            cur_path = []
            forward_action = HabitatSimActions.MOVE_FORWARD
            left_action = HabitatSimActions.TURN_LEFT
            right_action = HabitatSimActions.TURN_RIGHT
            foward_unit = (
                self._sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount
            )
            angle = math.degrees(angle)
            self.turn(angle)
            forward_step = int(distance // foward_unit)
            for k in range(forward_step):
                if k == forward_step - 1:
                    output = self._sim.step(forward_action)
                    cur_path.append(self.get_agent_info())
                    break
                else:
                    self._sim.step_without_obs(forward_action)
                    cur_path.append(self.get_agent_info())
                    if self._sim.previous_step_collided:
                        output = self._sim.step(forward_action)
                        break
            forward_step = forward_step - len(cur_path)
            if forward_step > 0:
                init_try_angle = random.choice([90, 270])
                self.turn(init_try_angle)
                if init_try_angle == 90:
                    turn_seqs = [
                        {"head_turns": [], "tail_turns": [right_action] * 3},
                        {"head_turns": [right_action], "tail_turns": [right_action] * 2},
                        {"head_turns": [right_action], "tail_turns": [right_action]},
                        {"head_turns": [right_action] * 2, "tail_turns": [left_action]},
                        {"head_turns": [right_action], "tail_turns": [left_action] * 2},
                        {"head_turns": [right_action], "tail_turns": [left_action] * 3},
                    ]
                elif init_try_angle == 270:
                    turn_seqs = [
                        {"head_turns": [], "tail_turns": [left_action] * 3},
                        {"head_turns": [left_action], "tail_turns": [left_action] * 2},
                        {"head_turns": [left_action], "tail_turns": [left_action]},
                        {"head_turns": [left_action] * 2, "tail_turns": [right_action]},
                        {"head_turns": [left_action], "tail_turns": [right_action] * 2},
                        {"head_turns": [left_action], "tail_turns": [right_action] * 3},
                    ]
                for turn_seq in turn_seqs:
                    for turn in turn_seq["head_turns"]:
                        self._sim.step_without_obs(turn)
                    prev_position = self._sim.get_agent_state().position
                    self._sim.step_without_obs(forward_action)
                    post_posiiton = self._sim.get_agent_state().position
                    if list(prev_position) != list(post_posiiton):
                        cur_path.append(self.get_agent_info())
                        for turn in turn_seq["tail_turns"]:
                            self._sim.step_without_obs(turn)
                        for k in range(forward_step):
                            if k == forward_step - 1:
                                output = self._sim.step(forward_action)
                                cur_path.append(self.get_agent_info())
                            else:
                                self._sim.step_without_obs(forward_action)
                                cur_path.append(self.get_agent_info())
                                if self._sim.previous_step_collided:
                                    output = self._sim.step(forward_action)
                                    break
                        break
            output["cur_path"] = cur_path
            return output
