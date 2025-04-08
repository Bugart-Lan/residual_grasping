import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)

from drivers import PositionController
from utils import AddActuatedFloatingSphere


def MakeTrajectory():
    sample_times = [0, 0.1, 2, 3]
    positions = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 1, 0]),
            np.array([0, 0.5, 1, 0, 2, 0]),
            np.array([0, -0.5, 1, 0, 0, 0]),
        ]
    )
    return PiecewisePolynomial.FirstOrderHold(sample_times, positions.T)


meshcat = StartMeshcat()
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.005)
AddActuatedFloatingSphere(plant)
plant.Finalize()
controller = builder.AddSystem(PositionController(plant, robot="point"))
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
traj = MakeTrajectory()
source = builder.AddSystem(TrajectorySource(traj))
# source = builder.AddSystem(ConstantVectorSource(np.array([0, 0, 1, 0, 0, 0])))
builder.Connect(source.get_output_port(0), controller.GetInputPort("position"))

MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
diagram = builder.Build()

simulator = Simulator(diagram)
meshcat.StartRecording()
simulator.AdvanceTo(traj.end_time())
meshcat.PublishRecording()

input("Press Enter to exit.")
