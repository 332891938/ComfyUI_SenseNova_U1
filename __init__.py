from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# Register neo_chat / neo_vision with transformers before any node loads a checkpoint.
import sys
from pathlib import Path

_sensenova_src = Path(__file__).resolve().parent / "SenseNova" / "src"
if str(_sensenova_src) not in sys.path:
    sys.path.insert(0, str(_sensenova_src))
import sensenova_u1  # noqa: F401, E402

from .SenseNova_node import SenseNova_SM_Model,  SenseNova_SM_Sampler
class SenseNova_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SenseNova_SM_Model,
            SenseNova_SM_Sampler,
        ]   

async def comfy_entrypoint() -> SenseNova_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return SenseNova_SM_Extension()


