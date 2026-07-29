import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cb_common import base_cfg, new_workdir, assert_services_present
from capsule_brain.runtime.bootstrap import build_application


async def main():
    wd = new_workdir("smoke")
    app = build_application(base_cfg(wd))
    await app.start()
    names = assert_services_present(app)
    print("services present:", names)
    conv = app.services.require("conversation")
    resp = await conv.respond(conversation_id="smoke-1", text="Reply with exactly the word PONG.")
    print("response:", repr(resp.text[:80]))
    print("model:", resp.model_name, "provider:", resp.provider, "latency:", resp.latency_ms)
    await app.stop()
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
