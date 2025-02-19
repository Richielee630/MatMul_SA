# 2025-02-17T22:46:48.666561200
import vitis

client = vitis.create_client()
client.set_workspace(path="MatMul_SA")

comp = client.get_component(name="MatMul_SA")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

comp.run(operation="CO_SIMULATION")

comp.run(operation="PACKAGE")

vitis.dispose()

