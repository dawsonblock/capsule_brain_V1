"""Test 8 store phase (process A): write a unique nonce to the persistent
MemoryService, then shut the application down completely."""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg
from capsule_brain.memory.models import MemoryType
from capsule_brain.runtime.bootstrap import build_application

WORKDIR = HERE / "stores" / "t8_persist"


async def main():
    nonce = os.environ["ORBIT_NONCE"]
    # Fresh workdir each run is fine ONLY if we also clear it; but we need the
    # same DB to persist across processes. Remove any prior db so the test is
    # deterministic.
    for f in WORKDIR.glob("*.sqlite"):
        f.unlink(missing_ok=True)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    app = build_application(base_cfg(WORKDIR))
    await app.start()
    memory = app.services.require("memory")
    rec = await memory.write(
        nonce,
        type=MemoryType.OBSERVATION,
        source="qualification",
        importance=1.0,
        protected=True,
        metadata={"qualification_nonce": True},
    )
    await app.stop()
    print(f"STORED:{nonce}")
    print(f"MEMORY_ID:{rec.id}")


if __name__ == "__main__":
    asyncio.run(main())
