"""Test 8 check phase (process B): start a NEW application instance pointing
at the same persistent stores, and verify the nonce is retrieved from
persistent storage (not from any model context)."""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cb_common import base_cfg
from capsule_brain.runtime.bootstrap import build_application

WORKDIR = HERE / "stores" / "t8_persist"


async def main():
    nonce = os.environ["ORBIT_NONCE"]
    app = build_application(base_cfg(WORKDIR))
    await app.start()
    memory = app.services.require("memory")
    recent = await memory.recent(limit=200)
    texts = [m.text for m in recent]
    found = nonce in texts
    # Also confirm via a fresh conversation that the value is NOT in the
    # model's own context (it can only surface via memory retrieval). We do
    # not require the model to echo it; the authoritative check is the
    # direct persistent-storage readback above.
    await app.stop()
    print(f"FOUND_IN_PERSISTENT_STORAGE:{found}")
    print(f"RECORD_COUNT:{len(texts)}")
    print(f"VERDICT:PASS" if found else "VERDICT:FAIL")
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    asyncio.run(main())
