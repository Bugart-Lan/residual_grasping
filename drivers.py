import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import Diagram, DiagramBuilder, LeafSystem
from pydrake.systems.primitives import (
    Demultiplexer,
    FirstOrderLowPassFilter,
    PassThrough,
    StateInterpolatorWithDiscreteDerivative,
)

kp = np.array([2000, 1500, 1500, 1500, 1500, 500, 500])
kd = 2 * np.sqrt(kp)
ki = np.ones_like(kp)

pid_coeffs = {
    "iiwa": {
        "kp": np.array([2000, 1500, 1500, 1500, 1500, 500, 500]),
    },
    "point": {
        "kp": 100 * np.ones((6, 1)),
    },
}
pid_coeffs["iiwa"]["kd"] = 2 * np.sqrt(pid_coeffs["iiwa"]["kp"])
pid_coeffs["iiwa"]["ki"] = np.ones_like(pid_coeffs["iiwa"]["kp"])
pid_coeffs["point"]["kd"] = 2 * np.sqrt(pid_coeffs["point"]["kp"])
pid_coeffs["point"]["ki"] = 20 * np.ones_like(pid_coeffs["point"]["kp"])


class GripperPoseToPosition(LeafSystem):

    def __init__(self, X_GB=RigidTransform()):
        LeafSystem.__init__(self)
        self.DeclareAbstractInputPort("X_WG", AbstractValue.Make(RigidTransform()))
        self.DeclareVectorOutputPort("position", 6, self.CalcPoseToPosition)
        self._X_GB = X_GB

    def CalcPoseToPosition(self, context, output):
        X_WG = self.get_input_port(0).Eval(context)
        X_WB = X_WG @ self._X_GB
        output.SetFromVector(
            np.concatenate(
                [
                    X_WB.translation(),
                    RollPitchYaw(X_WB.rotation()).vector()[::-1],
                ]
            )
        )


class PositionController(Diagram):

    def __init__(self, controller_plant, robot="iiwa"):
        Diagram.__init__(self)

        # Create a diagram
        builder = DiagramBuilder()
        num_positions = controller_plant.num_positions()

        # State -> positions, velocities
        state_demux = builder.AddNamedSystem(
            "demultiplexer", Demultiplexer(2 * num_positions, num_positions)
        )
        builder.ExportInput(state_demux.get_input_port(), "state")
        builder.ExportOutput(state_demux.get_output_port(0), "position_measured")
        builder.ExportOutput(state_demux.get_output_port(1), "velocity_estimated")

        # External contact forces
        contact_forces = builder.AddNamedSystem(
            "low_pass_filter", FirstOrderLowPassFilter(0.01, num_positions)
        )
        builder.ExportInput(
            contact_forces.get_input_port(), "generalized_contact_forces"
        )
        builder.ExportOutput(contact_forces.get_output_port(), "torque_external")

        # Estimate velocities from desired positions
        interpolator = builder.AddNamedSystem(
            "velocity_interpolator",
            StateInterpolatorWithDiscreteDerivative(num_positions, 0.01, True),
        )
        builder.ExportInput(interpolator.get_input_port(), "position")

        # State feedback control with PID to generate accelerations
        inverse_dynamics = builder.AddNamedSystem(
            "inverse_dynamics_controller",
            InverseDynamicsController(
                controller_plant,
                pid_coeffs[robot]["kp"],
                pid_coeffs[robot]["kd"],
                pid_coeffs[robot]["ki"],
                False,
            ),
        )
        builder.Connect(
            interpolator.GetOutputPort("state"),
            inverse_dynamics.GetInputPort("desired_state"),
        )
        builder.ConnectInput("state", inverse_dynamics.GetInputPort("estimated_state"))
        builder.ExportOutput(
            inverse_dynamics.GetOutputPort("generalized_force"), "actuation"
        )

        position_pass_through = builder.AddNamedSystem(
            "position_pass_through", PassThrough(num_positions)
        )
        builder.ConnectInput("position", position_pass_through.get_input_port())
        builder.ExportOutput(
            position_pass_through.get_output_port(), "position_commanded"
        )

        state_pass_through = builder.AddNamedSystem(
            "state_pass_through", PassThrough(2 * num_positions)
        )
        builder.ConnectInput("state", state_pass_through.get_input_port())
        builder.ExportOutput(state_pass_through.get_output_port(), "state_estimated")

        builder.BuildInto(self)


class PointFingerForceControl(LeafSystem):

    def __init__(self, plant, mass):
        LeafSystem.__init__(self)
        self._plant = plant
        self._mass = mass + plant.GetRigidBodyByName("sphere").default_mass()
        dof = plant.num_actuated_dofs()

        self.DeclareVectorInputPort("desired_contact_force", dof)
        self.DeclareVectorOutputPort("finger_actuation", dof, self.CalcOutput)

    def CalcOutput(self, context, output):
        g = self._plant.gravity_field().gravity_vector()
        desired_force = self.get_input_port(0).Eval(context)
        desired_force[:3] -= self._mass * g
        output.SetFromVector(desired_force)
