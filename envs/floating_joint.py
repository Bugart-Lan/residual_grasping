import gymnasium as gym
import numpy as np

from pydrake.common.value import AbstractValue
from pydrake.geometry import MeshcatVisualizer
from pydrake.gym import DrakeGymEnv
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.perception import PointCloud
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_graphviz
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.systems.primitives import ConstantVectorSource, Multiplexer, PassThrough
from pydrake.systems.sensors import Image, ImageRgba8U
from pydrake.visualization import AddFrameTriadIllustration
from manipulation.scenarios import AddRgbdSensor, AddRgbdSensors
from manipulation.utils import ConfigureParser

from drivers import PositionController
from utils import AddActuatedFloatingSphere


# Gym parameters.
sim_time_step = 0.01
gym_time_step = 0.05
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[0]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

gym.envs.register(
    id="FloatingJoint-v0", entry_point=("envs.floating_joint:DrakeFloatingJointEnv")
)


def make_sim(meshcat=None, time_limit=5, debug=False, obs_noise=False, mass=1):
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_approximation,
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    parser = Parser(plant)
    ConfigureParser(parser)
    # parser.AddModelsFromUrl("package://manipulation/camera_box.sdf")[0]
    # plant.WeldFrames(plant.GetFrameByName("world"), plant.GetFrameByName("base"))
    AddActuatedFloatingSphere(plant, mass=mass)
    plant.Finalize()
    rgbd = AddRgbdSensor(builder, scene_graph, RigidTransform())
    # AddRgbdSensors(builder, plant, scene_graph, model_instance_prefix="camera")
    AddFrameTriadIllustration(
        scene_graph=scene_graph, frame=plant.GetFrameByName("sphere")
    )

    # Controller plant
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddActuatedFloatingSphere(controller_plant, mass=mass)
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")

    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    if debug:
        for i in range(plant.num_model_instances()):
            print(
                f"Model Instance {i}: {plant.GetModelInstanceName(ModelInstanceIndex(i))}"
            )

        # Visualize the plant.
        import matplotlib.pyplot as plt

        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)

    # Command with position
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
    actions = builder.AddSystem(PassThrough(3))
    source = builder.AddSystem(ConstantVectorSource(np.array([0, 0, 0])))
    multiplexer = builder.AddSystem(Multiplexer([3, 3]))
    builder.Connect(actions.get_output_port(0), multiplexer.get_input_port(0))
    builder.Connect(source.get_output_port(0), multiplexer.get_input_port(1))
    builder.Connect(multiplexer.get_output_port(0), controller.GetInputPort("position"))
    builder.ExportInput(actions.get_input_port(0), "actions")

    class ObservationPublisher(LeafSystem):
        def __init__(self, noise=False):
            LeafSystem.__init__(self)
            self.ns = plant.num_multibody_states()
            self.DeclareVectorInputPort("plant_states", self.ns)
            self.DeclareAbstractInputPort("image", AbstractValue.Make(ImageRgba8U()))
            self.DeclareVectorOutputPort("observations", self.ns, self.CalcObs)
            self.noise = noise

        def CalcObs(self, context, output):
            plant_state = self.get_input_port(0).Eval(context)
            image = self.get_input_port(1).Eval(context)
            # height = image.height()
            # width = image.width()
            # size = image.size()
            # print(height, width, size)
            if self.noise:
                plant_state += np.random.uniform(low=-0.01, high=0.01, size=self.ns)
            output.set_value(plant_state)

    # sensor = builder.GetSubsystemByName("camera_box")
    obs_pub = builder.AddSystem(ObservationPublisher(noise=obs_noise))
    builder.Connect(plant.get_state_output_port(), obs_pub.get_input_port(0))
    builder.Connect(rgbd.get_output_port(0), obs_pub.get_input_port(1))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("state", 12)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            state = self.get_input_port(0).Eval(context)
            cost = np.linalg.norm(state[:3]) ** 2  # distance to the origin
            cost += 0.2 * np.linalg.norm(state[6:9]) ** 2  # sphere velocities
            output[0] = 5 - cost

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
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    if debug:
        import pydot

        pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png(
            "images/FloatingJoint-v0-diagram.png"
        )

    return simulator


def reset_handler(simulator, diagram_context, seed):
    np.random.seed(seed)
    diagram = simulator.get_system()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sphere = plant.GetModelInstanceByName("sphere")
    plant.SetPositions(
        plant_context,
        sphere,
        np.concatenate((np.random.random(3) * 0.5 - 1, np.zeros(3))),
    )
    plant.SetVelocities(plant_context, sphere, np.array([0, 0, 0, 0, 0, 0]))


def info_handler(simulator: Simulator) -> dict:
    info = dict()
    info["timestamp"] = simulator.get_context().get_time()
    return info


def DrakeFloatingJointEnv(
    meshcat=None, time_limit=gym_time_limit, debug=False, obs_noise=False
):
    simulator = make_sim(
        meshcat=meshcat, time_limit=time_limit, debug=debug, obs_noise=obs_noise
    )
    plant = simulator.get_system().GetSubsystemByName("plant")

    # Define action space
    action_space = gym.spaces.Box(
        low=np.asarray([-1, -1, -1]),
        high=np.asarray([1, 1, 1]),
        dtype=np.float64,
    )

    # Define observation space.
    low = np.concatenate(
        (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits())
    )
    high = np.concatenate(
        (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits())
    )

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
