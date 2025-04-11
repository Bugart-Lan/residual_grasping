import os
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    AddFrameTriadIllustration,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Quaternion,
    RigidTransform,
    RotationMatrix,
    SchunkWsgPositionController,
    Simulator,
    StartMeshcat,
    plot_system_graphviz,
)

from GraspSelector import GraspSelector
from GraspPlanner import GraspPlanner
from drivers import PositionController, GripperPoseToPosition

from utils import AddActuatedFloatingSphere, _ConfigureParser

from manipulation.scenarios import AddRgbdSensors


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

CAMERA_INSTANCE_PREFIX = "camera"


env_directive = "file://" + os.getcwd() + "/models/env.dmd.yaml"
scene_directive = "file://" + os.getcwd() + "/models/full.dmd.yaml"
scene_directive = "package://models/full.dmd.yaml"


meshcat = StartMeshcat()
input("Press Enter to continue...")


def load_scenario(directive, obj_name="sugar", rng=None):
    # Create a new diagram
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    _ConfigureParser(parser, include_manipulation=True)

    # Load object in random pose
    parser.AddModelsFromUrl(OBJECTS[obj_name]["url"])
    # Generate a random quaternion
    if not rng:
        rng = np.random.default_rng()
    q = rng.random(size=4)
    q /= np.linalg.norm(q)
    # Generate a random position
    x = rng.random() * 0.1 - 0.05
    y = rng.random() * 0.1 - 0.05
    z = rng.random() * 0.2 + 0.1
    plant.SetDefaultFreeBodyPose(
        plant.GetBodyByName(OBJECTS[obj_name]["base"]),
        RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z]),
    )

    # Load directive
    parser.AddModelsFromUrl(directive)
    # Add a floating joint and weld it to the wsg gripper
    sphere = AddActuatedFloatingSphere(plant)
    plant.WeldFrames(
        plant.GetFrameByName("sphere"),
        plant.GetFrameByName("body"),
        X_FM=RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]),
    )
    plant.Finalize()
    AddRgbdSensors(
        builder, plant, scene_graph, model_instance_prefix=CAMERA_INSTANCE_PREFIX
    )

    # Add controller for the gripper
    # Create a plant that contains only the floating sphere (joint)
    controller_plant = MultibodyPlant(0)
    # Controller for the floating joint
    AddActuatedFloatingSphere(controller_plant)
    controller_plant.Finalize()

    point_finger_position_control = builder.AddSystem(
        PositionController(controller_plant, "point")
    )
    builder.Connect(
        point_finger_position_control.GetOutputPort("actuation"),
        plant.GetInputPort("sphere_actuation"),
    )
    builder.Connect(
        plant.GetOutputPort("sphere_generalized_contact_forces"),
        point_finger_position_control.GetInputPort("generalized_contact_forces"),
    )
    builder.Connect(
        plant.GetOutputPort("sphere_state"),
        point_finger_position_control.GetInputPort("state"),
    )
    builder.ExportInput(
        point_finger_position_control.GetInputPort("position"), "gripper_pose"
    )

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

    AddFrameTriadIllustration(
        scene_graph=scene_graph, frame=plant.GetFrameByName("body")
    )
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    return builder.Build()


def main(obj_name: str = "sugar", show_diagram: bool = False, verbose: bool = False):
    rng = np.random.default_rng()
    meshcat.Delete()

    builder = DiagramBuilder()
    scenario = builder.AddSystem(
        load_scenario(scene_directive, obj_name=obj_name, rng=rng)
    )
    plant = scenario.GetSubsystemByName("plant")

    # Set initial position of the floating joint
    # TODO: debug the case where the gripper has initial rotation
    sphere = plant.GetModelInstanceByName("sphere")
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
    builder.Connect(
        transformer.get_output_port(0), scenario.GetInputPort("gripper_pose")
    )
    builder.Connect(
        planner.GetOutputPort("wsg_position"), scenario.GetInputPort("command")
    )

    diagram = builder.Build()

    if show_diagram:
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.show()

    meshcat.StartRecording()
    simulator = Simulator(diagram)
    simulator_context = simulator.get_context()
    simulator.AdvanceTo(5)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    plant_context = diagram.GetSubsystemContext(plant, simulator_context)
    body = plant.GetBodyByName(OBJECTS[obj_name]["base"])
    pose = plant.EvalBodyPoseInWorld(plant_context, body)
    if verbose:
        print(f"Object's final position {pose.translation()[2]}")
    return pose.translation()[2] >= 0.3


N = 1
cnt_success = 0
for i in range(N):
    print(f"Running {i}-th test...")
    results = main(obj_name="meat", show_diagram=False)
    cnt_success += results

print(f"# of successful grasp = {cnt_success}")
print(f"# of failed grasp = {N - cnt_success}")
print(f"Success rate = {cnt_success / N}")
input("Press Enter to exit.")
