import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    AddFrameTriadIllustration,
    AddMultibodyPlantSceneGraph,
    AngleAxis,
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SchunkWsgPositionController,
    Simulator,
    StartMeshcat,
    TrajectorySource,
    plot_system_graphviz,
)

from drivers import PositionController

from utils import AddActuatedFloatingSphere

meshcat = StartMeshcat()


def MakeGripperFrames(X_WG, X_WO):
    # Define the gripper pose relative to the object when in grasp
    p_GgraspO = [0, 0, -0.2]
    R_GgraspO = RotationMatrix.MakeZRotation(np.pi / 2)
    X_GgraspO = RigidTransform(R_GgraspO, p_GgraspO)
    X_OGgrasp = X_GgraspO.inverse()

    # Define pregrasp gripper pose
    X_GgraspGpregrasp = RigidTransform([0, 0, 0.1])

    X_WG["pick"] = X_WO["initial"] @ X_OGgrasp
    X_WG["prepick"] = X_WG["pick"] @ X_GgraspGpregrasp
    X_WG["place"] = X_WO["goal"] @ X_OGgrasp
    X_WG["preplace"] = X_WG["place"] @ X_GgraspGpregrasp

    X_GprepickGpreplace = X_WG["prepick"].inverse() @ X_WG["preplace"]
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GprepickGpreplace.translation() / 2.0 - [0, 0, -0.2],
    )
    X_WG["clearance"] = X_WG["prepick"] @ X_GprepickGclearance

    # Set the timing
    s = 2  # time needed per unit translation
    times = {"initial": 0}
    X_GinitialGprepick = X_WG["initial"].inverse() @ X_WG["prepick"]
    times["prepick"] = times["initial"] + s * np.linalg.norm(
        X_GinitialGprepick.translation()
    )
    times["pick_start"] = times["prepick"] + 1.0
    times["pick_end"] = times["pick_start"] + 1.0
    X_WG["pick_start"] = X_WG["pick"]
    X_WG["pick_end"] = X_WG["pick"]
    times["postpick"] = times["pick_end"] + 2.0
    X_WG["postpick"] = X_WG["prepick"]
    time_to_from_clearance = s * np.linalg.norm(X_GprepickGclearance.translation())
    times["clearance"] = times["postpick"] + time_to_from_clearance
    times["preplace"] = times["clearance"] + time_to_from_clearance
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 1.0
    X_WG["place_start"] = X_WG["place"]
    X_WG["place_end"] = X_WG["place"]
    times["postplace"] = times["place_end"] + 2.0
    X_WG["postplace"] = X_WG["preplace"]

    return X_WG, times


def MakeGripperPoseTrajectory(X_G, times):
    sample_times = []
    positions = []
    for name in [
        "initial",
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        "clearance",
        "preplace",
        "place_start",
        "place_end",
        "postplace",
    ]:
        sample_times.append(times[name])
        positions.append(
            np.concatenate(
                [
                    X_G[name].translation(),
                    RollPitchYaw(X_G[name].rotation()).vector()[::-1],
                ]
            )
        )

    return PiecewisePolynomial.FirstOrderHold(sample_times, np.array(positions).T)


def MakeGripperCommandTrajectory(times):
    opened = np.array([0.107])
    closed = np.array([0.0])
    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [
            times["initial"],
            times["pick_start"],
            times["pick_end"],
            times["place_start"],
            times["place_end"],
            times["postplace"],
        ],
        np.hstack([[opened], [opened], [closed], [closed], [opened], [opened]]),
    )

    return traj_wsg_command


def AddGripper(plant):
    parser = Parser(plant)
    wsg = parser.AddModels(
        url="package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
    )[0]
    plant.RenameModelInstance(wsg, "wsg")

    total_mass = 0
    for ind in plant.GetBodyIndices(wsg):
        total_mass += plant.get_body(ind).default_mass()

    return wsg, total_mass


def AddBrickAndFloor(plant):
    parser = Parser(plant)
    brick = parser.AddModels(
        url="package://drake_models/manipulation_station/061_foam_brick.sdf"
    )
    parser.AddModels("sdf/floor.sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("floor_frame"))

    return brick


def CreateScene(show_diagram=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.001)
    AddGripper(plant)
    AddActuatedFloatingSphere(plant)
    AddBrickAndFloor(plant)

    # Weld the sphere to the wsg gripper
    plant.WeldFrames(
        plant.GetFrameByName("sphere"),
        plant.GetFrameByName("body"),
        X_FM=RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2), [0, 0, -0.1]),
    )

    plant.Finalize()

    # Add controller for point finger
    controller_plant = MultibodyPlant(0)
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

    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    AddFrameTriadIllustration(
        scene_graph=scene_graph, frame=plant.GetFrameByName("base_link")
    )
    AddFrameTriadIllustration(
        scene_graph=scene_graph, frame=plant.GetFrameByName("sphere")
    )
    diagram = builder.Build()

    if show_diagram:
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.show()

    return diagram


X_O = {
    "initial": RigidTransform([0, 0, 0.01]),
    "goal": RigidTransform(RotationMatrix.MakeZRotation(np.pi / 3), [0.5, 0, 0]),
}
X_G = {"initial": RigidTransform(RollPitchYaw([0, np.pi / 2, 0]), [1, 0, 0.5])}

builder = DiagramBuilder()
constant_value_source = builder.AddSystem(
    ConstantVectorSource(np.array([1, 0.5, 1, np.pi / 2, 0, np.pi / 2]))
)
scene = builder.AddNamedSystem("scene", CreateScene(show_diagram=False))
scene_context = scene.CreateDefaultContext()


# Set initial sphere finger location
plant = scene.GetSubsystemByName("plant")
sphere_finger = plant.GetModelInstanceByName("sphere")
plant_context = plant.GetMyContextFromRoot(scene_context)
plant.SetDefaultPositions(
    sphere_finger,
    np.concatenate(
        [
            X_G["initial"].translation(),
            RollPitchYaw(X_G["initial"].rotation()).vector()[::-1],
        ]
    ),
)
plant.SetDefaultFreeBodyPose(plant.GetBodyByName("base_link"), X_O["initial"])

X_G, times = MakeGripperFrames(X_G, X_O)
traj = MakeGripperPoseTrajectory(X_G, times)
X_G_source = builder.AddNamedSystem("traj_source", TrajectorySource(traj))
builder.Connect(X_G_source.get_output_port(), scene.GetInputPort("gripper_pose"))

traj_wsg_command = MakeGripperCommandTrajectory(times)
wsg_source = builder.AddSystem(TrajectorySource(traj_wsg_command))
builder.Connect(wsg_source.get_output_port(), scene.GetInputPort("command"))

diagram = builder.Build()

simulator = Simulator(diagram)
context = simulator.get_mutable_context()

diagram.ForcedPublish(context)
meshcat.StartRecording()
simulator.AdvanceTo(traj.end_time())
meshcat.PublishRecording()

input("Press Enter to finish.")
