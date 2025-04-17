import numpy as np

from enum import Enum

from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import LeafSystem
from pydrake.trajectories import PiecewisePose, PiecewisePolynomial


def MakeGripperFrames(X_WG, t0=0):
    # Define pregrasp gripper pose
    X_GgraspGpregrasp = RigidTransform([0, -0.1, 0])

    X_WG["prepick"] = X_WG["pick"] @ X_GgraspGpregrasp

    # Set the timing
    times = {"initial": t0}
    X_GinitialGprepick = X_WG["initial"].inverse() @ X_WG["prepick"]
    times["prepick"] = times["initial"] + 2 * np.linalg.norm(
        X_GinitialGprepick.translation()
    )
    times["pick_start"] = times["prepick"] + 1.0
    times["pick_end"] = times["pick_start"] + 1.0
    X_WG["pick_start"] = X_WG["pick"]
    X_WG["pick_end"] = X_WG["pick"]
    times["postpick"] = times["pick_end"] + 1.0
    X_WG["postpick"] = RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, 0.7])
    times["end"] = times["postpick"] + 10.0
    X_WG["end"] = X_WG["postpick"]

    return X_WG, times


def MakeGripperPoseTrajectory(X_G, times):
    sample_times = []
    poses = []
    for name in [
        "initial",
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        "end",
    ]:
        sample_times.append(times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


def MakeGripperCommandTrajectory(times):
    opened = np.array([0.107])
    closed = np.array([0.0])
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [
            times["initial"],
            times["pick_start"],
            times["pick_end"],
            times["postpick"] + 5,
        ],
        np.hstack([[opened], [opened], [closed], [closed]]),
    )

    return traj_wsg_command


class PlannerState(Enum):
    WAIT = 1
    PICK = 2


class GraspPlanner(LeafSystem):

    def __init__(self, plant):
        LeafSystem.__init__(self)
        # Inputs
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self._grasp_index = self.DeclareAbstractInputPort(
            "grasp", AbstractValue.Make((np.inf, RigidTransform()))
        ).get_index()
        # self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()
        # Outputs
        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcGripperPose,
        )
        self.DeclareVectorOutputPort("wsg_position", 1, self.CalcWsgPosition)
        # States
        self._gripper_body_index = plant.GetBodyByName("body").index()
        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(PlannerState.WAIT)
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._traj_X_G_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )
        self._traj_wsg_index = self.DeclareAbstractState(
            AbstractValue.Make(PiecewisePolynomial())
        )

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._mode_index)).get_value()
        times = context.get_abstract_state(int(self._times_index)).get_value()
        if mode == PlannerState.WAIT:
            # Wait for some amount of time for the objects to settle down
            if context.get_time() - times["initial"] > 1.0:
                self.Plan(context, state)
            return

    def Plan(self, context, state):
        X_G = {
            "initial": self.get_input_port(0).Eval(context)[
                int(self._gripper_body_index)
            ]
        }

        cost, X_G["pick"] = self.get_input_port(self._grasp_index).Eval(context)
        # TODO: deal with the case where the selector could not find a valid grasp
        # assert not np.isinf(cost), "Could not find a valid grasp."
        state.get_mutable_abstract_state(int(self._mode_index)).set_value(
            PlannerState.PICK
        )
        X_G["place"] = RigidTransform([0.5, 0.5, 0.3])
        X_G, times = MakeGripperFrames(X_G, t0=context.get_time())
        print(
            f"Planned {times['postpick'] - times['initial']} s trajectory at time {context.get_time()}"
        )
        traj_X_G = MakeGripperPoseTrajectory(X_G, times)
        traj_wsg_command = MakeGripperCommandTrajectory(times)

        state.get_mutable_abstract_state(int(self._traj_X_G_index)).set_value(traj_X_G)
        state.get_mutable_abstract_state(int(self._traj_wsg_index)).set_value(
            traj_wsg_command
        )

    def CalcGripperPose(self, context, output):
        traj_X_G = context.get_abstract_state(int(self._traj_X_G_index)).get_value()
        if traj_X_G.get_number_of_segments() > 0 and traj_X_G.is_time_in_range(
            context.get_time()
        ):
            output.set_value(traj_X_G.GetPose(context.get_time()))
            return
        output.set_value(
            self.get_input_port(0).Eval(context)[int(self._gripper_body_index)]
        )

    def CalcWsgPosition(self, context, output):
        opened = np.array([0.107])

        traj_wsg = context.get_abstract_state(int(self._traj_wsg_index)).get_value()
        if traj_wsg.get_number_of_segments() > 0 and traj_wsg.is_time_in_range(
            context.get_time()
        ):
            output.set_value(traj_wsg.value(context.get_time()))
            return
        output.SetFromVector([opened])
