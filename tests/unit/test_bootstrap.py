import pytest

from capsule_brain.runtime.bootstrap import build_application


@pytest.mark.asyncio
async def test_bootstrap_starts_goal_planner_after_event_bus(tmp_path):
    app = build_application({
        "goal_planner": {"db_path": str(tmp_path / "goals.sqlite")},
        "redis_bridge": {"enable": False},
    })

    await app.start()

    health = await app.services.health()
    assert health["event_bus"].healthy
    assert health["goal_planner"].healthy

    await app.stop()


def test_build_application_rejects_smoke_workflow_without_acknowledgement():
    """Construction must fail closed when smoke success is not acknowledged."""
    with pytest.raises(ValueError, match="allow_smoke_success"):
        build_application(
            {
                "workflow": {
                    "enable": True,
                    "verification_mode": "smoke",
                },
            }
        )
