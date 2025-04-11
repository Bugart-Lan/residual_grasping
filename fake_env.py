import numpy as np

from pydrake.all import (
    AddMultibodyPlant,
    AddMultibodyPlantSceneGraph,
    ApplyVisualizationConfig,
    DiagramBuilder,
    MeshcatVisualizer,
    MultibodyPlantConfig,
    Parser,
    Quaternion,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    VisualizationConfig,
)

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

sim_time_step = 0.005
obj_name = "sugar"
drake_contact_models = ["point", "hydroelastic_with_fallback"]
contact_model = drake_contact_models[0]
drake_contact_approximations = ["sap", "tamsi", "similar", "lagged"]
contact_approximation = drake_contact_approximations[0]

meshcat = StartMeshcat()
input("Press Enter to continue.")


def make_scene(plant=None, builder=None, obj_name="sugar"):
    parser = Parser(builder=builder, plant=plant)
    _ConfigureParser(parser, include_manipulation=True)
    parser.AddModelsFromUrl("package://models/full.dmd.yaml")
    return parser.AddModelsFromUrl(OBJECTS[obj_name]["url"])[0]


builder = DiagramBuilder()
multibody_plant_config = MultibodyPlantConfig(
    time_step=sim_time_step,
    contact_model=contact_model,
    discrete_contact_approximation=contact_approximation,
)
plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
# plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
obj = make_scene(builder=builder, obj_name=obj_name)
plant.SetDefaultFreeBodyPose(plant.GetBodyByName("body"), RigidTransform([0, 0, 1]))

# Randomize object pose
rng = np.random.default_rng()
q = rng.random(size=4)
q /= np.linalg.norm(q)
x = rng.random() * 0.1 - 0.05
y = rng.random() * 0.1 - 0.05
z = rng.random() * 0.2 + 2
plant.SetDefaultFreeBodyPose(
    plant.GetBodyByName(OBJECTS[obj_name]["base"]),
    RigidTransform(RotationMatrix(Quaternion(q)), [x, y, z]),
)
plant.Finalize()

MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
config = VisualizationConfig()
config.publish_contacts = True
config.publish_proximity = True
ApplyVisualizationConfig(config, builder, meshcat=meshcat)
diagram = builder.Build()
simulator = Simulator(diagram)
# simulator.Initialize()
# context = simulator.get_context()
meshcat.StartRecording()
simulator.AdvanceTo(1)
meshcat.PublishRecording()
