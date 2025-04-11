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
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_graphviz
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
from pydrake.systems.primitives import Adder, Demultiplexer, Saturation
from pydrake.visualization import AddFrameTriadIllustration
from manipulation.scenarios import AddRgbdSensors


from drivers import GripperPoseToPosition, PositionController
from utils import AddActuatedFloatingSphere, _ConfigureParser
from GraspSelector import GraspSelector
from GraspPlanner import GraspPlanner

# Objects
OBJECTS = {
    "sugar": {
        "base": "base_link_sugar",
        "url": "package://manipulation/hydro/004_sugar_box.sdf",
    },
    "soup": {
        "base": "base_link_soup",
        "url": "package://manipulation/hydro/005_tomato_soup_can.sdf",
    },
    "mustard": {
        "base": "base_link_mustard",
        "url": "package://manipulation/hydro/006_mustard_bottle.sdf",
    },
    "gelatin": {
        "base": "base_link_gelatin",
        "url": "package://manipulation/hydro/009_gelatin_box.sdf",
    },
    "meat": {
        "base": "base_link_meat",
        "url": "package://manipulation/hydro/010_potted_meat_can.sdf",
    },
}
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
CAMERA_INSTANCE_PREFIX = "camera"

# Gym parameters
sim_time_step = 0.001
gym_time_step = 0.05
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[1]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

gym.envs.register(
    id="ResidualGrasp-v0", entry_point=("envs.residual_grasp:DrakeResidualGraspEnv")
)


def load_all_objects(plant, parser, active="sugar", rng=None):
    active_obj = None
    for key, val in OBJECTS.items():
        if key == active:
            active_obj = parser.AddModelsFromUrl(val["url"])[0]
            if not rng:
                rng = np.random.default_rng()
            q = rng.random(size=4)
            q /= np.linalg.norm(q)
            x = rng.random() * 0.1 - 0.05
            y = rng.random() * 0.1 - 0.05
            z = rng.random() * 0.2 + 0.1
            plant.SetDefaultFreeBodyPose(
                plant.GetBodyByName(val["base"]),
                RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z]),
            )
        else:
            parser.AddModelsFromUrl(val["url"])
            plant.SetDefaultFreeBodyPose(
                plant.GetBodyByName(val["base"]),
                RigidTransform([1, 1, 0.1]),
            )
    return active_obj


def load_scenario(meshcat=None, obj_name="sugar", rng=None):
    # Create a new diagram
    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_approximation,
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
    obj = load_all_objects(plant, parser, active=obj_name)
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
    builder.ExportOutput(plant.get_state_output_port(obj), "object_state")

    diagram = builder.Build()
    diagram.set_name("env")

    return diagram


def make_sim(meshcat=None, time_limit=5, debug=False, obs_noise=False):
    rng = np.random.default_rng()
    obj_name = list(OBJECTS.keys())[np.random.randint(0, len(OBJECTS))]
    meshcat.Delete()

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
            meshcat=meshcat,
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

    planner = builder.AddSystem(GraspPlanner(plant))
    transformer = builder.AddSystem(
        GripperPoseToPosition(
            X_GB=RigidTransform(
                RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]
            ).inverse()
        )
    )
    builder.Connect(
        grasp_selector.GetOutputPort("grasp_selection"), planner.GetInputPort("grasp")
    )
    builder.Connect(
        scenario.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(planner.GetOutputPort("X_WG"), transformer.get_input_port(0))

    # Actions from RL agent
    demux = builder.AddSystem(Demultiplexer([6, 1]))
    builder.ExportInput(demux.get_input_port(0), "actions")
    adder = builder.AddSystem(Adder(2, 6))
    builder.Connect(demux.get_output_port(0), adder.get_input_port(0))
    builder.Connect(transformer.get_output_port(0), adder.get_input_port(1))
    builder.Connect(adder.get_output_port(0), scenario.GetInputPort("position"))
    adder = builder.AddSystem(Adder(2, 1))
    builder.Connect(demux.get_output_port(1), adder.get_input_port(0))
    builder.Connect(planner.GetOutputPort("wsg_position"), adder.get_input_port(1))
    sat = builder.AddSystem(Saturation(np.array([0]), np.array([0.107])))
    builder.Connect(adder.get_output_port(0), sat.get_input_port(0))
    builder.Connect(sat.get_output_port(0), scenario.GetInputPort("command"))

    class ObservationPublisher(LeafSystem):
        def __init__(self, noise=False):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("sphere_state", 12)
            self.DeclareVectorInputPort("wsg_state", 4)
            self.DeclareAbstractInputPort("image0", AbstractValue.Make(ImageRgba8U()))
            self.DeclareAbstractInputPort("image1", AbstractValue.Make(ImageRgba8U()))
            self.DeclareAbstractInputPort("image2", AbstractValue.Make(ImageRgba8U()))

            self.DeclareAbstractOutputPort(
                "observations",
                lambda: AbstractValue.Make(
                    {
                        "state": np.zeros(16),
                        "image0": np.zeros((width, height, 4)),
                        "image1": np.zeros((width, height, 4)),
                        "image2": np.zeros((width, height, 4)),
                    }
                ),
                self.CalcObs,
            )
            self.noise = noise

        def CalcObs(self, context, output):
            sphere_state = self.get_input_port(0).Eval(context)
            wsg_state = self.get_input_port(1).Eval(context)
            image0 = self.get_input_port(2).Eval(context)
            image1 = self.get_input_port(3).Eval(context)
            image2 = self.get_input_port(4).Eval(context)

            if self.noise:
                sphere_state += np.random.uniform(
                    low=-0.01, high=0.01, size=sphere_state.shape
                )
                wsg_state += np.random.uniform(
                    low=-0.01, high=0.01, size=wsg_state.shape
                )
                image0 += np.random.uniform(low=-0.01, high=0.01, size=image0.shape)
                image1 += np.random.uniform(low=-0.01, high=0.01, size=image1.shape)
                image2 += np.random.uniform(low=-0.01, high=0.01, size=image2.shape)

            output.set_value(
                {
                    "state": np.concatenate([sphere_state, wsg_state]),
                    "image0": image0.data,
                    "image1": image1.data,
                    "image2": image2.data,
                }
            )

    obs_pub = builder.AddSystem(ObservationPublisher(noise=obs_noise))
    builder.Connect(
        scenario.GetOutputPort("sphere_state"),
        obs_pub.get_input_port(0),
    )
    builder.Connect(
        scenario.GetOutputPort("wsg_state"),
        obs_pub.get_input_port(1),
    )
    builder.Connect(
        scenario.GetOutputPort("camera0_rgb_image"), obs_pub.get_input_port(2)
    )
    builder.Connect(
        scenario.GetOutputPort("camera1_rgb_image"), obs_pub.get_input_port(3)
    )
    builder.Connect(
        scenario.GetOutputPort("camera1_rgb_image"), obs_pub.get_input_port(4)
    )
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("object_state", 13)
            self.DeclareVectorInputPort("gripper_state", 12)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            object_state = self.get_input_port(0).Eval(context)
            gripper_state = self.get_input_port(1).Eval(context)
            cost = np.linalg.norm(gripper_state[6:9]) + np.linalg.norm(
                gripper_state[9:]
            )
            reward = 5 if object_state[2] >= 0.3 else 0
            output[0] = reward - cost

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(
        scenario.GetOutputPort("object_state"),
        reward.get_input_port(0),
    )
    builder.Connect(
        scenario.GetOutputPort("sphere_state"),
        reward.get_input_port(1),
    )
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    def monitor(context):
        scenario_context = scenario.GetMyContextFromRoot(context)
        obj_state = scenario.GetOutputPort("object_state").Eval(scenario_context)

        if obj_state[2] < 0:
            print("object falls below 0")
            return EventStatus.ReachedTermination(diagram, "object falls below 0")
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    if debug:
        import pydot

        pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png(
            "images/ResidualGrasp-v0-diagram.png"
        )

    return simulator


def reset_handler(simulator, diagram_context, seed):
    np.random.seed(seed)
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
                np.random.random(2) * 0.1 - 0.2,
                np.random.random(1) * 0.3 + 0.5,
                np.ones(3) * 0,
            ]
        ),
    )
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetPositions(plant_context, wsg, np.array([0, 0]))


def info_handler(simulator: Simulator) -> dict:
    info = dict()
    info["timestamp"] = simulator.get_context().get_time()
    return info


def DrakeResidualGraspEnv(
    meshcat=None, time_limit=gym_time_limit, debug=False, obs_noise=False
):
    simulator = make_sim(
        meshcat=meshcat, time_limit=time_limit, debug=debug, obs_noise=obs_noise
    )
    plant = simulator.get_system().GetSubsystemByName("env").GetSubsystemByName("plant")
    if debug:
        names = plant.GetPositionNames()
        lim_low = plant.GetPositionLowerLimits()
        lim_high = plant.GetPositionUpperLimits()
        for i in range(len(names)):
            print(f"{names[i]}: low={lim_low[i]}, high={lim_high[i]}")

    # Define action space
    action_space = gym.spaces.Box(
        low=np.asarray([-0.1, -0.1, 0.1, -0.1, -0.1, -0.1, -0.107]) * 0.01,
        high=np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.107]) * 0.01,
        dtype=np.float64,
    )

    # Define observation space
    tol = 1e-3
    low = [-1.0] * 3 + [-np.inf] * 9 + [-0.055, -tol, -np.inf, -np.inf]
    high = [1.0] * 3 + [np.inf] * 9 + [tol, 0.055, np.inf, np.inf]

    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(
                low=np.asarray(low), high=np.asarray(high), dtype=np.float64
            ),
            "image0": gym.spaces.Box(
                low=0,
                high=255,
                shape=(height, width, channel),
                dtype=np.uint8,
            ),
            "image1": gym.spaces.Box(
                low=0,
                high=255,
                shape=(height, width, channel),
                dtype=np.uint8,
            ),
            "image2": gym.spaces.Box(
                low=0,
                high=255,
                shape=(height, width, channel),
                dtype=np.uint8,
            ),
        }
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
