import gymnasium as gym
import numpy as np
import os

from pydrake.common.value import AbstractValue
from pydrake.geometry import MeshcatVisualizer
from pydrake.gym import DrakeGymEnv
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_graphviz
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.systems.sensors import ImageRgba8U


from drivers import PositionController
from utils import AddActuatedFloatingSphere, _ConfigureParser

# Gym parameters.
sim_time_step = 0.01
gym_time_step = 0.05
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[0]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

gym.envs.register(id="Grasp-v0", entry_point=("envs.grasp:DrakeGraspEnv"))


def make_scene(plant=None, builder=None):
    url = "file://" + os.getcwd() + "/models/full.dmd.yaml"
    parser = Parser(builder=builder, plant=plant)
    _ConfigureParser(parser, include_manipulation=True)
    parser.AddModelsFromUrl(url)


def make_sim(meshcat=None, time_limit=5, debug=False, obs_noise=False):
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_approximation,
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    make_scene(builder=builder)
    AddActuatedFloatingSphere(plant)
    plant.WeldFrames(
        plant.GetFrameByName("sphere"),
        plant.GetFrameByName("body"),
        X_FM=RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]),
    )
    plant.Finalize()

    # Controller plant
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddActuatedFloatingSphere(controller_plant)
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")

    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Extract the controller plant information.
    ns = controller_plant.num_multibody_states()
    nv = controller_plant.num_velocities()
    na = controller_plant.num_actuators()
    nj = controller_plant.num_joints()
    npos = controller_plant.num_positions()

    if debug:
        for i in range(plant.num_model_instances()):
            print(
                f"Model Instance {i}: {plant.GetModelInstanceName(ModelInstanceIndex(i))}"
            )

        print(
            f"\nNumber of position: {npos},",
            f"Number of velocities: {nv},",
            f"Number of actuators: {na},",
            f"Number of joints: {nj},",
            f"Number of multibody states: {ns}",
        )

        # Visualize the plant.
        import matplotlib.pyplot as plt

        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)

    # Using positions as the action space
    controller = builder.AddSystem(
        PositionController(controller_plant=controller_plant, robot="point")
    )
    builder.Connect(
        controller.GetOutputPort("actuation"),
        plant.GetInputPort("sphere_actuation"),
    )
    builder.Connect(
        plant.GetOutputPort("sphere_generalized_contact_forces"),
        controller.GetInputPort("generalized_contact_forces"),
    )
    builder.Connect(
        plant.GetOutputPort("sphere_state"),
        controller.GetInputPort("state"),
    )
    builder.ExportInput(controller.GetInputPort("position"), "actions")

    class ObservationPublisher(LeafSystem):
        def __init__(self, noise=False):
            LeafSystem.__init__(self)
            self.ns = plant.num_multibody_states()
            self.DeclareVectorInputPort("plant_states", self.ns)
            self.DeclareAbstractInputPort(
                "point_cloud", AbstractValue.Make(ImageRgba8U())
            )
            self.DeclareAbstractOutputPort(
                "observations", lambda: AbstractValue.Make(ImageRgba8U()), self.CalcObs
            )
            self.DeclareVectorOutputPort("observations", self.ns, self.CalcObs)
            self.noise = noise

        def CalcObs(self, context, output):
            plant_state = self.get_input_port(0).Eval(context)
            if self.noise:
                plant_state += np.random.uniform(low=-0.01, high=0.01, size=self.ns)
            output.set_value(plant_state)

    sensor = builder.GetSubsystemByName("camera_box")
    obs_pub = builder.AddSystem(ObservationPublisher(noise=obs_noise))
    builder.Connect(plant.get_state_output_port(), obs_pub.get_input_port(0))
    builder.Connect(sensor.get_output_port(0), obs_pub.get_input_port(1))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            # ns = plant.num_multibody_states()
            self.DeclareVectorInputPort("state", 12)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            # TODO: Configure reward
            reward = 1
            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(
        plant.get_state_output_port(plant.GetModelInstanceByName("sphere")),
        reward.get_input_port(0),
    )
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    def monitor(context):
        plant_context = plant.GetMyContextFromRoot(context)
        state = plant.GetOutputPort("state").Eval(plant_context)

        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")

    simulator.set_monitor(monitor)

    return simulator


def reset_handler(simulator, diagram_context, seed):
    np.random.seed(seed)
    home_positions = []
    diagram = simulator.get_system()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sphere = plant.GetModelInstanceByName("sphere")
    plant.SetPositions(plant_context, sphere, np.array([0, 0, 0.7, 0, 0, 0]))


def info_handler(simulator: Simulator) -> dict:
    info = dict()
    info["timestamp"] = simulator.get_context().get_time()
    return info


def DrakeGraspEnv(
    meshcat=None, time_limit=gym_time_limit, debug=False, obs_noise=False
):
    simulator = make_sim(
        meshcat=meshcat, time_limit=time_limit, debug=debug, obs_noise=obs_noise
    )
    plant = simulator.get_system().GetSubsystemByName("plant")

    # Define action space
    action_space = gym.spaces.Box(
        low=np.asarray([-2 * np.pi, -2 * np.pi, -2 * np.pi, -1, -1, -1]),
        high=np.asarray([2 * np.pi, 2 * np.pi, 2 * np.pi, 1, 1, 1]),
        dtype=np.float64,
    )

    # Define observation space.
    low = np.concatenate(
        (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits())
    )
    high = np.concatenate(
        (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits())
    )
    print(low)
    print(high)
    observation_space = gym.spaces.Box(
        low=np.asarray(low), high=np.asarray(high), dtype=np.float64
    )

    env = DrakeGymEnv(
        simulator=simulator,
        time_step=gym_time_step,
        action_space=action_space,
        observation_space=observation_space,
        reward="reward",
        action_port_id="actions",
        observation_port_id="observations",
        reset_handler=reset_handler,
        info_handler=info_handler,
    )

    return env
