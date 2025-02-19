# 2025-02-17T22:43:05.954761700
import vitis

client = vitis.create_client()
client.set_workspace(path="MatMul_SA")

comp = client.create_hls_component(name = "MatMul_SA",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

comp = client.get_component(name="MatMul_SA")
comp.run(operation="C_SIMULATION")

comp.run(operation="SYNTHESIS")

