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
from pydrake.systems.analysis import Simulator
from pydrake.systems.drawing import plot_graphviz
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.systems.sensors import CameraInfo, ImageRgba8U
from pydrake.systems.primitives import Demultiplexer
from manipulation.scenarios import AddRgbdSensors


from drivers import PositionController
from utils import AddActuatedFloatingSphere, _ConfigureParser

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
height_threshold = 0.3 # z-coord higher than this threshold is considered as a success

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
sim_time_step = 0.01
gym_time_step = 0.05
controller_time_step = 0.01
gym_time_limit = 5
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[0]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

gym.envs.register(
    id="EndToEndGrasp-v0", entry_point=("envs.end_to_end_grasp:DrakeEndToEndGraspEnv")
)


def make_scene(plant=None, builder=None, obj_name="sugar"):
    parser = Parser(builder=builder, plant=plant)
    _ConfigureParser(parser, include_manipulation=True)
    parser.AddModelsFromUrl("package://models/full.dmd.yaml")
    return parser.AddModelsFromUrl(OBJECTS[obj_name]["url"])[0]


def make_sim(meshcat=None, time_limit=5, debug=False, obs_noise=False):
    # TODO: randomize objects
    obj_name = "sugar"

    builder = DiagramBuilder()
    multibody_plant_config = MultibodyPlantConfig(
        time_step=sim_time_step,
        contact_model=contact_model,
        discrete_contact_approximation=contact_approximation,
    )
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    obj = make_scene(builder=builder, obj_name=obj_name)
    AddActuatedFloatingSphere(plant)
    plant.WeldFrames(
        plant.GetFrameByName("sphere"),
        plant.GetFrameByName("body"),
        X_FM=RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]),
    )
    # Randomize object pose
    rng = np.random.default_rng()
    q = rng.random(size=4)
    q /= np.linalg.norm(q)
    x = rng.random() * 0.1 - 0.05
    y = rng.random() * 0.1 - 0.05
    z = rng.random() * 0.2 + 0.1
    plant.SetDefaultFreeBodyPose(
        plant.GetBodyByName(OBJECTS[obj_name]["base"]),
        RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z]),
    )
    plant.Finalize()

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

    # Define actions = [sphere_positions, gripper_state] = [(6,), (1,)]
    demux = builder.AddSystem(Demultiplexer([6, 1]))
    builder.ExportInput(demux.get_input_port(0), "actions")

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
    builder.Connect(demux.get_output_port(0), controller.GetInputPort("position"))

    # Add controller for the wsg gripper
    wsg_driver = builder.AddSystem(SchunkWsgPositionController())
    builder.Connect(
        wsg_driver.GetOutputPort("generalized_force"),
        plant.GetInputPort("wsg_actuation"),
    )
    builder.Connect(plant.GetOutputPort("wsg_state"), wsg_driver.GetInputPort("state"))
    builder.Connect(
        demux.get_output_port(1), wsg_driver.GetInputPort("desired_position")
    )

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
                sphere_state += np.random.uniform(low=-0.01, high=0.01, size=sphere_state.shape)
                wsg_state += np.random.uniform(low=-0.01, high=0.01, size=wsg_state.shape)
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
    builder.Connect(plant.get_state_output_port(plant.GetModelInstanceByName("sphere")), obs_pub.get_input_port(0))
    builder.Connect(plant.get_state_output_port(plant.GetModelInstanceByName("wsg")), obs_pub.get_input_port(1))
    builder.Connect(camera0.get_output_port(0), obs_pub.get_input_port(2))
    builder.Connect(camera1.get_output_port(0), obs_pub.get_input_port(3))
    builder.Connect(camera2.get_output_port(0), obs_pub.get_input_port(4))
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
            cost = np.linalg.norm(gripper_state[6:9]) + np.linalg.norm(gripper_state[9:])
            reward = 1 if object_state[2] >= 0.3 else 0
            output[0] = reward - cost

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(
        plant.get_state_output_port(obj),
        reward.get_input_port(0),
    )
    builder.Connect(
        plant.get_state_output_port(plant.GetModelInstanceByName("sphere")),
        reward.get_input_port(1),
    )
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    def monitor(context):
        plant_context = plant.GetMyContextFromRoot(context)
        wsg_state = plant.GetOutputPort("wsg_state").Eval(plant_context)
        sphere_state = plant.GetOutputPort("sphere_state").Eval(plant_context)

        if wsg_state[0] <= -0.055 or wsg_state[0] >= 0:
            return EventStatus.ReachedTermination(diagram, "wsg left position exceeds limit")
        
        if wsg_state[1] <= 0 or wsg_state[0] >= 0.055:
            return EventStatus.ReachedTermination(diagram, "wsg right position exceeds limit")

        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    if debug:
        import pydot

        pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].write_png(
            "images/EndToEndGrasp-v0-diagram.png"
        )

    return simulator


def reset_handler(simulator, diagram_context, seed):
    np.random.seed(seed)
    diagram = simulator.get_system()
    plant = diagram.GetSubsystemByName("plant")
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    sphere = plant.GetModelInstanceByName("sphere")
    plant.SetPositions(plant_context, sphere, np.random.random(6) * 0.1 + 0.1)
    wsg = plant.GetModelInstanceByName("wsg")
    plant.SetPositions(plant_context, wsg, np.array([0, 0]))


def info_handler(simulator: Simulator) -> dict:
    info = dict()
    info["timestamp"] = simulator.get_context().get_time()
    return info


def DrakeEndToEndGraspEnv(
    meshcat=None, time_limit=gym_time_limit, debug=False, obs_noise=False
):
    simulator = make_sim(
        meshcat=meshcat, time_limit=time_limit, debug=debug, obs_noise=obs_noise
    )
    plant = simulator.get_system().GetSubsystemByName("plant")

    # Define action space
    action_space = gym.spaces.Box(
        low=np.asarray([-1, -1, -1, -1, -1, -1, 0]),
        high=np.asarray([1, 1, 1, 1, 1, 1, 0.107]),
        dtype=np.float64,
    )

    # Define observation space.
    low = [-1.] * 6 + [-np.inf] * 6 + [-0.055, 0., -np.inf, -np.inf]
    high = [1.] * 6 + [np.inf] * 6 + [0., 0.055, np.inf, np.inf]


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
