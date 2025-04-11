import numpy as np
import os

from pydrake.geometry import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    MeshcatVisualizer,
    ProximityProperties,
    Sphere,
    StartMeshcat,
)
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import (
    PrismaticJoint,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


def AddSphere(plant, shape, name, mass=1.0, mu=1.0, color=[0.5, 0.5, 0.9, 1.0]):
    instance = plant.AddModelInstance(name)
    inertia = UnitInertia.SolidSphere(shape.radius())
    body = plant.AddRigidBody(
        name,
        instance,
        SpatialInertia(mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia),
    )
    if plant.geometry_source_is_registered():
        proximity_properties = ProximityProperties()
        AddContactMaterial(1e4, 1e7, CoulombFriction(mu, mu), proximity_properties)
        AddCompliantHydroelasticProperties(0.01, 1e8, proximity_properties)
        plant.RegisterCollisionGeometry(
            body, RigidTransform(), shape, name, proximity_properties
        )

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)
    return instance


def AddActuatedFloatingSphere(plant, mass=1000.0):
    fingers = [
        {
            "name": "finger_x",
            "axis": [1, 0, 0],
            "type": PrismaticJoint,
        },
        {
            "name": "finger_y",
            "axis": [0, 1, 0],
            "type": PrismaticJoint,
        },
        {
            "name": "finger_z",
            "axis": [0, 0, 1],
            "type": PrismaticJoint,
        },
        {
            "name": "finger_rz",
            "axis": [0, 0, 1],
            "type": RevoluteJoint,
        },
        {
            "name": "finger_ry",
            "axis": [0, 1, 0],
            "type": RevoluteJoint,
        },
        {
            "name": "finger_rx",
            "axis": [1, 0, 0],
            "type": RevoluteJoint,
        },
    ]

    sphere = AddSphere(plant, Sphere(0.05), "sphere", mass=mass)

    curr_frame = plant.world_frame()
    for i in range(6):
        rigidbody = plant.AddRigidBody(
            f"false_body{i + 1}",
            sphere,
            SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
        )
        finger_i = plant.AddJoint(
            fingers[i]["type"](
                fingers[i]["name"],
                curr_frame,
                rigidbody.body_frame(),
                fingers[i]["axis"],
                -1 if i < 3 else -np.inf,
                1 if i < 3 else np.inf,
            )
        )
        plant.AddJointActuator(fingers[i]["name"], finger_i)
        curr_frame = rigidbody.body_frame()
    plant.WeldFrames(curr_frame, plant.GetFrameByName("sphere"))

    return sphere


def _ConfigureParser(parser: Parser, include_manipulation=False):
    package_map = parser.package_map()
    package_xml = os.path.join(os.path.dirname(__file__), "models/package.xml")
    package_map.AddPackageXml(filename=package_xml)
    if include_manipulation:
        from manipulation.utils import ConfigureParser

        ConfigureParser(parser)


if __name__ == "__main__":
    meshcat = StartMeshcat()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.1)
    AddActuatedFloatingSphere(plant)
    plant.Finalize()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    print(plant.num_actuated_dofs())

    diagram = builder.Build()
    simulator = Simulator(diagram)

    simulator.AdvanceTo(1)

    input("Press Enter to exit.")
