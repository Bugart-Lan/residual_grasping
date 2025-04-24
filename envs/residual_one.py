from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import warnings

from pydrake.common import RandomGenerator
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
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    MultibodyPlant,
    MultibodyPlantConfig,
)
from pydrake.perception import PointCloud
from pydrake.systems.analysis import Simulator, SimulatorStatus
from pydrake.systems.drawing import plot_graphviz
from pydrake.systems.framework import Context, DiagramBuilder, EventStatus, LeafSystem
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
from pydrake.systems.primitives import ConstantVectorSource, PassThrough
from pydrake.visualization import AddFrameTriadIllustration
from manipulation.scenarios import AddRgbdSensors

from drivers import GripperPoseToPosition, PositionController
from utils import AddActuatedFloatingSphere, _ConfigureParser, PointCloudMerger, Switch
from GraspSelector import GraspSelector
from GraspPlanner import GraspPlanner


class SE3Adder(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("residual", 7)  # [quaternion xyz] from actions
        self.DeclareAbstractInputPort(
            "grasp_selction", AbstractValue.Make((np.inf, RigidTransform()))
        )
        self.DeclareAbstractOutputPort(
            "X_WG",
            lambda: AbstractValue.Make((np.inf, RigidTransform())),
            self.CalcOutput,
        )

    def CalcOutput(self, context, output):
        r = self.get_input_port(0).Eval(context)
        print("residual =", r)
        cost, transform = self.get_input_port(1).Eval(context)
        if not np.isfinite(cost):
            transform = RigidTransform.Identity()
        q = transform.rotation().ToQuaternion().wxyz() + r[:4]
        q = (
            q / np.linalg.norm(q)
            if np.linalg.norm(q) > 1e-6
            else np.array([1, 0, 0, 0])
        )
        print("t =", transform.translation())
        t = transform.translation() + r[4:]
        print("dt =", r[4:])
        print("t =", t)
        t[:2] = np.clip(t[:2], -0.4, 0.4)
        t[2] = np.clip(t[2], 0.0, 0.25)
        output.set_value((0, RigidTransform(Quaternion(q), t)))


# Objects
OBJECTS = {
    "sugar": {
        "id": 0,
        "name": "004_sugar_box",
        "base": "base_link_sugar",
        "url": "package://drake_models/ycb/004_sugar_box.sdf",
    },
    "soup": {
        "id": 1,
        "name": "005_tomato_soup_can",
        "base": "base_link_soup",
        "url": "package://drake_models/ycb/005_tomato_soup_can.sdf",
    },
    "mustard": {
        "id": 2,
        "name": "006_mustard_bottle",
        "base": "base_link_mustard",
        "url": "package://drake_models/ycb/006_mustard_bottle.sdf",
    },
    "gelatin": {
        "id": 3,
        "name": "009_gelatin_box",
        "base": "base_link_gelatin",
        "url": "package://drake_models/ycb/009_gelatin_box.sdf",
    },
    "meat": {
        "id": 4,
        "name": "010_potted_meat_can",
        "base": "base_link_meat",
        "url": "package://drake_models/ycb/010_potted_meat_can.sdf",
    },
}

t_graspend = 2.1
t_end = 3.1
height_threshold = 0.25  # z-coord higher than this threshold is considered as a success

# Camera parameters
width = 80
height = 80
channel = 4
foy_y = np.pi / 4.0
near = 0.1
far = 10.0
renderer = "my_renderer"
image_size = width * height * 4
CAMERA_INSTANCE_PREFIX = "camera"
cloud_size = 400

# Gym parameters
sim_time_step = 0.005
gym_time_step = 0.1
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[0]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

gym.envs.register(
    id="ResidualGraspOne-v0",
    entry_point=("envs.residual_one:DrakeResidualGraspOneStepEnv"),
)


def reset_all_objects(plant, context=None, active="sugar", rng=None):
    for key, val in OBJECTS.items():
        if key == active:
            if not rng:
                rng = np.random.default_rng()
            q = rng.random(size=4)
            q = (
                q / np.linalg.norm(q)
                if np.linalg.norm(q) > 1e-6
                else np.array([1, 0, 0, 0])
            )
            x = rng.random() * 0.1 - 0.05
            y = rng.random() * 0.1 - 0.05
            z = rng.random() * 0.1 + 0.15
            transform = RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z])
        else:
            # TODO: Make it don't start at the same location
            transform = RigidTransform([1, 1, -1] + rng.random(3) - 1)

        if plant.is_finalized():
            plant.SetFreeBodyPose(context, plant.GetBodyByName(val["base"]), transform)
        else:  # Hide inactive objects to below the floor
            plant.SetDefaultFreeBodyPose(plant.GetBodyByName(val["base"]), transform)


def load_scenario(meshcat=None, obj_name="sugar", rng=None):
    # Create a new diagram
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_approximation,
        penetration_allowance=1e-4,
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    parser = Parser(plant)
    _ConfigureParser(parser, include_manipulation=True)
    parser.AddModelsFromUrl("package://models/full.dmd.yaml")
    # Add a floating joint and weld it to the wsg gripper
    sphere = AddActuatedFloatingSphere(plant)
    plant.WeldFrames(
        plant.GetFrameByName("sphere"),
        plant.GetFrameByName("body"),
        X_FM=RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]),
    )
    for key, val in OBJECTS.items():
        parser.AddModelsFromUrl(val["url"])
    reset_all_objects(plant, active=obj_name, rng=rng)
    plant.Finalize()
    plant.SetDefaultPositions(
        sphere,
        np.concatenate(
            [
                np.random.random(2) * 0.1 - 0.2,
                np.random.random(1) * 0.3 + 0.5,
                np.ones(3) * 0,
            ]
        ),
    )
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
        also_add_point_clouds=True,
        model_instance_prefix="camera",
        depth_camera=depth_camera,
        renderer=renderer,
    )

    # Create a plant that contains only the floating sphere (joint)
    controller_plant = MultibodyPlant(0)
    AddActuatedFloatingSphere(controller_plant)
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")

    if meshcat:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        AddFrameTriadIllustration(
            scene_graph=scene_graph, frame=plant.GetFrameByName("sphere")
        )
        AddFrameTriadIllustration(
            scene_graph=scene_graph, frame=plant.GetFrameByName("body")
        )

    controller = builder.AddSystem(PositionController(controller_plant, "point"))
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
    builder.ExportInput(controller.GetInputPort("position"), "position")

    # Add controller for the wsg gripper
    wsg_driver = builder.AddSystem(SchunkWsgPositionController())
    builder.Connect(
        wsg_driver.GetOutputPort("generalized_force"),
        plant.GetInputPort("wsg_actuation"),
    )
    builder.Connect(plant.GetOutputPort("wsg_state"), wsg_driver.GetInputPort("state"))
    builder.ExportInput(wsg_driver.GetInputPort("desired_position"), "command")

    builder.ExportOutput(plant.GetOutputPort("body_poses"), "body_poses")
    builder.ExportOutput(plant.GetOutputPort("wsg_state"), "wsg_state")
    builder.ExportOutput(plant.GetOutputPort("sphere_state"), "sphere_state")

    switch = builder.AddSystem(Switch(len(OBJECTS), 13))
    selector = builder.AddNamedSystem(
        "selector", ConstantVectorSource([OBJECTS[obj_name]["id"]])
    )
    builder.ExportOutput(selector.get_output_port(0), "active_obj_index")
    builder.Connect(selector.get_output_port(0), switch.get_input_port(0))
    for key, val in OBJECTS.items():
        builder.Connect(
            plant.GetOutputPort(f"{val['name']}_state"),
            switch.get_input_port(val["id"] + 1),
        )
    builder.ExportOutput(switch.get_output_port(0), "object_state")
    diagram = builder.Build()
    diagram.set_name("env")

    return diagram


def make_sim(meshcat=None, time_limit=5, wait_time=0.5, debug=False, obs_noise=False):
    rng = np.random.default_rng()
    obj_name = list(OBJECTS.keys())[np.random.randint(0, len(OBJECTS))]

    builder = DiagramBuilder()
    scenario = builder.AddSystem(
        load_scenario(meshcat=meshcat, obj_name=obj_name, rng=rng)
    )
    plant = scenario.GetSubsystemByName("plant")
    grasp_selector = builder.AddSystem(
        GraspSelector(
            [
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
            ],
            meshcat=meshcat if debug else None,
            noise=obs_noise,
        )
    )
    builder.Connect(
        scenario.GetOutputPort("camera0_point_cloud"),
        grasp_selector.GetInputPort("cloud0_W"),
    )
    builder.Connect(
        scenario.GetOutputPort("camera1_point_cloud"),
        grasp_selector.GetInputPort("cloud1_W"),
    )
    builder.Connect(
        scenario.GetOutputPort("camera2_point_cloud"),
        grasp_selector.GetInputPort("cloud2_W"),
    )
    builder.Connect(
        scenario.GetOutputPort("body_poses"), grasp_selector.GetInputPort("body_poses")
    )

    planner = builder.AddSystem(GraspPlanner(plant, wait_time=wait_time))
    builder.Connect(
        scenario.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )

    actions = builder.AddSystem(PassThrough(7))
    builder.ExportInput(actions.get_input_port(0), "actions")
    adder = builder.AddSystem(SE3Adder())
    builder.Connect(actions.get_output_port(0), adder.get_input_port(0))
    builder.Connect(grasp_selector.get_output_port(0), adder.get_input_port(1))
    builder.Connect(adder.get_output_port(0), planner.GetInputPort("grasp"))
    pose_to_position = builder.AddSystem(
        GripperPoseToPosition(
            X_GB=RigidTransform(
                RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]
            ).inverse()
        )
    )
    builder.Connect(planner.GetOutputPort("X_WG"), pose_to_position.get_input_port(0))
    builder.Connect(
        pose_to_position.get_output_port(0), scenario.GetInputPort("position")
    )
    builder.Connect(
        planner.GetOutputPort("wsg_position"), scenario.GetInputPort("command")
    )

    class ObservationPublisher(LeafSystem):
        def __init__(self, noise=False):
            LeafSystem.__init__(self)
            self.DeclareAbstractInputPort(
                "grasp", AbstractValue.Make((np.inf, RigidTransform()))
            )
            self.DeclareAbstractInputPort("cloud", AbstractValue.Make(PointCloud()))
            self.DeclareVectorOutputPort(
                "observations", 7 + 3 * cloud_size, self.CalcObs
            )
            self.noise = noise

        def CalcObs(self, context, output):
            time = context.get_time()
            if t_graspend > time >= wait_time:
                cost, grasp = self.get_input_port(0).Eval(context)
                q = grasp.rotation().ToQuaternion().wxyz()
                t = grasp.translation()
                cloud = self.get_input_port(1).Eval(context)
                if self.noise:
                    points = cloud.mutable_xyzs()
                    points += np.array([[0.05], [0], [0]])
                if debug and meshcat:
                    meshcat.SetObject("observations/cloud", cloud, point_size=0.003)
                cloud.resize(cloud_size)
                output.SetFromVector(
                    np.concatenate([q, t, np.nan_to_num(cloud.xyzs().reshape(-1))])
                )
            else:
                output.SetFromVector(np.zeros(7 + 3 * cloud_size))

    obs_pub = builder.AddSystem(ObservationPublisher(noise=obs_noise))
    builder.Connect(grasp_selector.get_output_port(0), obs_pub.get_input_port(0))
    merger = builder.AddSystem(
        PointCloudMerger(noise=obs_noise, meshcat=meshcat if debug else None)
    )
    builder.Connect(
        scenario.GetOutputPort("camera0_point_cloud"), merger.get_input_port(0)
    )
    builder.Connect(
        scenario.GetOutputPort("camera1_point_cloud"), merger.get_input_port(1)
    )
    builder.Connect(
        scenario.GetOutputPort("camera2_point_cloud"), merger.get_input_port(2)
    )
    builder.Connect(merger.get_output_port(0), obs_pub.get_input_port(1))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("object_state", 13)
            self.DeclareVectorInputPort("gripper_state", 12)
            self.DeclareVectorInputPort("actions", 7)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            time = context.get_time()
            object_state = self.get_input_port(0).Eval(context)
            gripper_state = self.get_input_port(1).Eval(context)
            actions = self.get_input_port(2).Eval(context)
            print("reward actions =", actions)
            z = RotationMatrix(RollPitchYaw(gripper_state[5:2:-1])).matrix()[:, 2]
            tar = gripper_state[0:3] - 0.2 * z
            cost = np.linalg.norm(actions)
            reward = 10 if object_state[6] > height_threshold else 1
            if time < t_end:
                cost += np.linalg.norm(tar - object_state[4:7])
            if time > wait_time:
                output[0] = reward - cost
            else:
                output[0] = 0

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(
        scenario.GetOutputPort("object_state"),
        reward.get_input_port(0),
    )
    builder.Connect(
        scenario.GetOutputPort("sphere_state"),
        reward.get_input_port(1),
    )
    builder.Connect(actions.get_output_port(0), reward.get_input_port(2))
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
            "images/ResidualGraspOne-v0-diagram.png"
        )

    return simulator


def reset_handler(simulator, diagram_context, seed):
    rng = np.random.default_rng(seed=seed)
    diagram = simulator.get_system()
    env = diagram.GetSubsystemByName("env")
    env_context = diagram.GetSubsystemContext(env, diagram_context)
    plant = env.GetSubsystemByName("plant")
    plant_context = env.GetMutableSubsystemContext(plant, env_context)
    sphere = plant.GetModelInstanceByName("sphere")
    plant.SetPositions(
        plant_context,
        sphere,
        np.concatenate(
            [
                rng.random(2) * 0.1 - 0.2,
                rng.random(1) * 0.3 + 0.5,
                np.ones(3) * 0,
            ]
        ),
    )
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetPositions(plant_context, wsg, np.array([0, 0]))
    active_object = list(OBJECTS.keys())[np.random.randint(0, len(OBJECTS))]
    selector = env.GetSubsystemByName("selector")
    selector_context = env.GetMutableSubsystemContext(selector, env_context)
    selector.get_mutable_source_value(selector_context).set_value(
        [OBJECTS[active_object]["id"]]
    )
    reset_all_objects(plant, context=plant_context, active=active_object, rng=rng)


def info_handler(simulator: Simulator) -> dict:
    info = dict()
    info["timestamp"] = simulator.get_context().get_time()
    return info


class CustomDrakeGymEnv(DrakeGymEnv):
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
        wait_time: float = 0.5,
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
        self._wait_time = wait_time

    def step(self, action):
        context = self.simulator.get_context()
        time = context.get_time()
        print(f"Action @ t = {time}: {action}")
        self.action_port.FixValue(context, action)
        truncated = False

        prev_observation = self.observation_port.Eval(context)
        total_reward = 0
        try:
            if time < self._wait_time:
                status = self.simulator.AdvanceTo(self._wait_time)
            else:
                status = self.simulator.AdvanceTo(t_graspend)
                total_reward += self.reward(self.simulator.get_system(), context)
                status = self.simulator.AdvanceTo(t_end)
                total_reward += self.reward(self.simulator.get_system(), context)
        except RuntimeError as e:
            warnings.warn("Calling Done after catching RuntimeError")
            warnings.warn(e.args[0])
            truncated = True
            terminated = False
            reward = 0
            info = dict()

            return prev_observation, reward, terminated, truncated, info

        observation = self.observation_port.Eval(context)
        terminated = not truncated and (
            status.reason() == SimulatorStatus.ReturnReason.kReachedTerminationCondition
        )
        info = self.info_handler(self.simulator)
        return observation, total_reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        observations, info = super().reset(seed=seed, options=options)
        try:
            context = self.simulator.get_context()
            self.action_port.FixValue(context, np.zeros(7))
            self.simulator.AdvanceTo(self._wait_time)
        except RuntimeError as e:
            warnings.warn("Calling Done after catching RuntimeError (reset)")
            warnings.warn(e.args[0])
            return observations, dict()

        context = self.simulator.get_mutable_context()
        observations = self.observation_port.Eval(context)
        info = self.info_handler(self.simulator)

        return observations, info


def DrakeResidualGraspOneStepEnv(
    meshcat=None, time_limit=gym_time_limit, debug=False, obs_noise=False
):
    wait_time = 0.8
    simulator = make_sim(
        meshcat=meshcat,
        time_limit=time_limit,
        wait_time=wait_time,
        debug=debug,
        obs_noise=obs_noise,
    )

    # Define action space
    action_space = gym.spaces.Box(
        low=np.asarray([-0.01, -0.01, -0.01, -0.01, -0.1, -0.1, -0.1]),
        high=np.asarray([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1]),
        dtype=np.float64,
    )

    # Define observation space
    observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(7 + cloud_size * 3,), dtype=np.float64
    )

    env = CustomDrakeGymEnv(
        simulator=simulator,
        time_step=gym_time_step,
        action_space=action_space,
        observation_space=observation_space,
        reward="reward",
        action_port_id="actions",
        observation_port_id="observations",
        reset_handler=reset_handler,
        info_handler=info_handler,
        wait_time=wait_time,
    )

    return env
