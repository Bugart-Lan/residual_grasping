from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from pydrake.common.eigen_geometry import Quaternion
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    MeshcatVisualizer,
    RenderCameraCore,
)
from pydrake.gym import DrakeGymEnv
from pydrake.manipulation import SchunkWsgPositionController
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.analysis import Simulator, SimulatorStatus
from pydrake.systems.drawing import plot_graphviz
from pydrake.systems.framework import (
    BasicVector_,
    Context,
    DiagramBuilder,
    EventStatus,
    LeafSystem,
)
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
from pydrake.systems.primitives import Demultiplexer, PassThrough
from manipulation.scenarios import AddRgbdSensors


from drivers import GripperPoseToPosition, PositionController
from utils import AddActuatedFloatingSphere, _ConfigureParser

from GraspPlanner import GraspPlanner


class ActionToSE3(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("actions", 7)
        self.DeclareAbstractOutputPort(
            "grasp", lambda: AbstractValue.Make((0, RigidTransform())), self.CalcOutput
        )

    def CalcOutput(self, context, output):
        x = self.get_input_port(0).Eval(context)
        q = x[:4] / np.linalg.norm(x[:4])
        output.set_value((0, RigidTransform(Quaternion(q), x[4:])))


height_threshold = 0.3  # z-coord higher than this threshold is considered as a success

# Camera parameters
width = 640
height = 480
channel = 4
foy_y = np.pi / 4.0
near = 0.1
far = 10.0
renderer = "my_renderer"
image_size = width * height * 4

# Gym parameters
sim_time_step = 0.001
gym_time_step = 0.05
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[1]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

obj_name = "sugar"

gripper_transform = RigidTransform(
    RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]
)


def make_scene(plant=None, builder=None):
    parser = Parser(builder=builder, plant=plant)
    _ConfigureParser(parser, include_manipulation=True)
    parser.AddModelsFromUrl("package://models/full.dmd.yaml")
    return parser.AddModelsFromUrl("package://manipulation/hydro/004_sugar_box.sdf")[0]


def make_sim(meshcat=None, time_limit=5, debug=False, obs_noise=False):
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_approximation,
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    obj = make_scene(builder=builder)
    sphere = AddActuatedFloatingSphere(plant)
    plant.WeldFrames(
        plant.GetFrameByName("sphere"),
        plant.GetFrameByName("body"),
        X_FM=gripper_transform,
    )

    # Randomize object pose
    rng = np.random.default_rng()
    q = rng.random(size=4)
    q /= np.linalg.norm(q)
    x = rng.random() * 0.1 - 0.05
    y = rng.random() * 0.1 - 0.05
    z = rng.random() * 0.2 + 0.2
    plant.SetDefaultFreeBodyPose(
        plant.GetBodyByName("base_link_sugar"),
        RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z]),
    )
    plant.Finalize()

    plant.SetDefaultPositions(sphere, np.array([0, 0, 0.7, 0, 0, 0]))

    # Add cameras
    depth_camera = DepthRenderCamera(
        RenderCameraCore(
            renderer,
            CameraInfo(width=width, height=height, fov_y=foy_y),
            ClippingRange(near=near, far=far),
            RigidTransform(),
        ),
        DepthRange(near, far),
    )
    AddRgbdSensors(
        builder,
        plant,
        scene_graph,
        also_add_point_clouds=False,
        model_instance_prefix="camera",
        depth_camera=depth_camera,
        renderer=renderer,
    )
    camera0 = builder.GetSubsystemByName("camera0")
    camera1 = builder.GetSubsystemByName("camera1")
    camera2 = builder.GetSubsystemByName("camera2")

    # Controller plant
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddActuatedFloatingSphere(controller_plant)
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")

    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Define actions as the SE3 representation of the grasp pose
    # 7 dim: quaternion  + xyz translation
    actions = builder.AddSystem(ActionToSE3())
    builder.ExportInput(actions.get_input_port(0), "actions")

    planner = builder.AddSystem(GraspPlanner(plant))
    transformer = builder.AddSystem(
        GripperPoseToPosition(X_GB=gripper_transform.inverse())
    )
    builder.Connect(actions.get_output_port(0), planner.GetInputPort("grasp"))
    builder.Connect(
        plant.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(planner.GetOutputPort("X_WG"), transformer.get_input_port(0))

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
    builder.Connect(transformer.get_output_port(0), controller.GetInputPort("position"))

    # Add controller for the wsg gripper
    wsg_driver = builder.AddSystem(SchunkWsgPositionController())
    builder.Connect(
        wsg_driver.GetOutputPort("generalized_force"),
        plant.GetInputPort("wsg_actuation"),
    )
    builder.Connect(plant.GetOutputPort("wsg_state"), wsg_driver.GetInputPort("state"))
    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        wsg_driver.GetInputPort("desired_position"),
    )

    class ObservationPublisher(LeafSystem):
        def __init__(self, noise=False):
            LeafSystem.__init__(self)
            self.DeclareAbstractInputPort("image0", AbstractValue.Make(ImageRgba8U()))
            self.DeclareVectorOutputPort(
                "observations", width * height * channel, self.CalcObs
            )
            self.noise = noise

        def CalcObs(self, context, output):
            image0 = self.get_input_port(0).Eval(context)
            output.SetFromVector(image0.data.reshape(-1))

    obs_pub = builder.AddSystem(ObservationPublisher(noise=obs_noise))
    builder.Connect(camera0.get_output_port(0), obs_pub.get_input_port(0))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("object_state", 13)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            object_state = self.get_input_port(0).Eval(context)
            output[0] = 10 if object_state[6] >= 0.3 else 0.1

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(
        plant.get_state_output_port(obj),
        reward.get_input_port(0),
    )
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    def monitor(context):
        plant_context = plant.GetMyContextFromRoot(context)
        obj_state = plant.GetOutputPort("004_sugar_box_state").Eval(plant_context)
        # print(f"object z-position = {obj_state[2]}")

        if obj_state[6] < -0.01:
            print("object falls below 0")
            return EventStatus.ReachedTermination(diagram, "object falls below 0")
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    if debug:
        import pydot

        pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png(
            "images/OneStepEnd2EndGrasp-v0-diagram.png"
        )

    return simulator


def reset_handler(simulator, diagram_context, seed):
    rng = np.random.default_rng(seed)
    diagram = simulator.get_system()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sphere = plant.GetModelInstanceByName("sphere")
    plant.SetPositions(
        plant_context,
        sphere,
        np.concatenate(
            [
                rng.random(2) * 0.1 - 0.2,
                rng.random(1) * 0.3 + 0.1,
                np.zeros(3),
            ]
        ),
    )
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetPositions(plant_context, wsg, np.array([0, 0]))

    q = rng.random(size=4)
    q /= np.linalg.norm(q)
    x = rng.random() * 0.1 - 0.05
    y = rng.random() * 0.1 - 0.05
    z = rng.random() * 0.2 + 0.2
    plant.SetFreeBodyPose(
        plant_context,
        plant.GetBodyByName("base_link_sugar"),
        RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z]),
    )


def info_handler(simulator: Simulator) -> dict:
    info = dict()
    info["timestamp"] = simulator.get_context().get_time()
    return info


class OneStepEnd2EndGrasp(DrakeGymEnv):
    def __init__(
        self,
        simulator: Simulator,
        time_step: float,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        reward: str,
        action_port_id: str = None,
        observation_port_id: str = None,
        reset_handler: Callable[[Simulator, Context], None] = None,
        info_handler: Callable[[Simulator, Context], dict] = None,
    ):
        super().__init__(
            simulator=simulator,
            time_step=time_step,
            action_space=action_space,
            observation_space=observation_space,
            reward=reward,
            action_port_id=action_port_id,
            observation_port_id=observation_port_id,
            reset_handler=reset_handler,
            info_handler=info_handler,
        )

    def step(self, action):
        context = self.simulator.get_context()
        time = context.get_time()
        self.action_port.FixValue(context, action)
        if time < 1:
            status = self.simulator.AdvanceTo(1)
        else:
            status = self.simulator.AdvanceTo(time + self.step)

        truncated = False
        observation = self.observation_port.Eval(context)
        reward = self.reward(self.simulator.get_system(), context)
        terminated = not truncated and (
            status.reason() == SimulatorStatus.ReturnReason.kReachedTerminationCondition
        )
        info = self.info_handler(self.simulator)

        return observation.astype(np.uint8), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        context = self.simulator.get_mutable_context()
        context.SetTime(0)
        self.simulator.Initialize()
        self.simulator.get_system().SetDefaultContext(context)
        self.reset_handler(self.simulator, context, seed)
        observations = self.observation_port.Eval(context)
        info = self.info_handler(self.simulator)

        return observations.astype(np.uint8), info


gym.envs.register(
    id="OneStepEnd2EndGrasp-v0", entry_point=("envs.one_step_end2end_grasp:make_env")
)


def make_env(meshcat=None, time_limit=gym_time_limit, debug=False, obs_noise=False):
    simulator = make_sim(
        meshcat=meshcat, time_limit=time_limit, debug=debug, obs_noise=obs_noise
    )

    # Define action space
    action_space = gym.spaces.Box(
        low=np.asarray([-1, -1, -1, -1, -1, -1, -1]),
        high=np.asarray([1, 1, 1, 1, 1, 1, 1]),
        dtype=np.float64,
    )

    # Define observation space
    observation_space = gym.spaces.Box(
        low=0, high=255, shape=(height * width * channel,), dtype=np.uint8
    )

    env = OneStepEnd2EndGrasp(
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
