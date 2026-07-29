"""Immutable, versioned policy registry.

Policies are immutable artifacts. The active policy is never overwritten in
place. A candidate policy is stored with a manifest describing its lineage,
training data digest, split manifest digest, hyperparameters, metrics, and
promotion status. Promotion flips a pointer; it does not mutate the artifact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capsule_brain.autolearn.policy import LearnedPolicy

PROMOTION_STATUS_PENDING = "pending"
PROMOTION_STATUS_REJECTED = "rejected"
PROMOTION_STATUS_ACTIVE = "active"
PROMOTION_STATUS_SHADOW = "shadow"
PROMOTION_STATUS_ARCHIVED = "archived"


@dataclass(slots=True)
class PolicyManifest:
    policy_id: str
    policy_version: str
    policy_type: str
    feature_schema_version: str
    training_data_digest: str
    split_manifest_digest: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_timestamp: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    parent_policy: str = ""
    promotion_status: str = PROMOTION_STATUS_PENDING
    promotion_timestamp: str = ""
    promotion_reason: str = ""
    artifact_path: str = ""
    artifact_sha256: str = ""
    manifest_path: str = ""
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyManifest":
        return cls(
            policy_id=str(data.get("policy_id", "")),
            policy_version=str(data.get("policy_version", "")),
            policy_type=str(data.get("policy_type", "")),
            feature_schema_version=str(data.get("feature_schema_version", "")),
            training_data_digest=str(data.get("training_data_digest", "")),
            split_manifest_digest=str(data.get("split_manifest_digest", "")),
            hyperparameters=dict(data.get("hyperparameters", {}) or {}),
            training_timestamp=str(data.get("training_timestamp", "")),
            metrics=dict(data.get("metrics", {}) or {}),
            parent_policy=str(data.get("parent_policy", "")),
            promotion_status=str(data.get("promotion_status", PROMOTION_STATUS_PENDING)),
            promotion_timestamp=str(data.get("promotion_timestamp", "")),
            promotion_reason=str(data.get("promotion_reason", "")),
            artifact_path=str(data.get("artifact_path", "")),
            artifact_sha256=str(data.get("artifact_sha256", "")),
            manifest_path=str(data.get("manifest_path", "")),
            manifest_sha256=str(data.get("manifest_sha256", "")),
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


class PolicyRegistry:
    """Filesystem-backed registry of immutable policy artifacts.

    Layout:
        <root>/
          active.json          -> {"policy_id": "..."} pointer
          policies/
            <policy_id>.json   -> policy artifact (LearnedPolicy.to_dict())
            <policy_id>.manifest.json
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.policies_dir = self.root / "policies"
        self.active_pointer = self.root / "active.json"

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.policies_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, policy_id: str) -> Path:
        return self.policies_dir / f"{policy_id}.json"

    def manifest_path(self, policy_id: str) -> Path:
        return self.policies_dir / f"{policy_id}.manifest.json"

    def register(
        self,
        policy: LearnedPolicy,
        *,
        training_data_digest: str,
        split_manifest_digest: str,
        hyperparameters: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        parent_policy: str = "",
        promotion_status: str = PROMOTION_STATUS_PENDING,
    ) -> PolicyManifest:
        """Persist a policy as an immutable artifact.

        Raises if a policy with the same id already exists — artifacts are
        write-once.
        """
        self._ensure_dirs()
        if not policy.policy_id:
            raise ValueError("policy.policy_id must be non-empty")
        art = self.artifact_path(policy.policy_id)
        man = self.manifest_path(policy.policy_id)
        if art.exists() or man.exists():
            raise FileExistsError(
                f"policy already registered: {policy.policy_id}"
            )

        artifact_text = policy.to_json()
        artifact_sha = _sha256_text(artifact_text)
        art.write_text(artifact_text)

        manifest = PolicyManifest(
            policy_id=policy.policy_id,
            policy_version=policy.policy_version,
            policy_type=policy.policy_type,
            feature_schema_version=policy.feature_schema_version,
            training_data_digest=training_data_digest,
            split_manifest_digest=split_manifest_digest,
            hyperparameters=dict(hyperparameters or {}),
            training_timestamp=policy.trained_at or _utc_now_iso(),
            metrics=dict(metrics or {}),
            parent_policy=parent_policy,
            promotion_status=promotion_status,
            artifact_path=str(art),
            artifact_sha256=artifact_sha,
            manifest_path=str(man),
        )
        manifest_text = json.dumps(manifest.to_dict(), sort_keys=True)
        manifest.manifest_sha256 = _sha256_text(manifest_text)
        man.write_text(manifest_text)
        return manifest

    def load_manifest(self, policy_id: str) -> PolicyManifest:
        man = self.manifest_path(policy_id)
        if not man.exists():
            raise FileNotFoundError(f"no manifest for policy {policy_id}")
        return PolicyManifest.from_dict(json.loads(man.read_text()))

    def load_policy(self, policy_id: str) -> LearnedPolicy:
        art = self.artifact_path(policy_id)
        if not art.exists():
            raise FileNotFoundError(f"no artifact for policy {policy_id}")
        return LearnedPolicy.from_json(art.read_text())

    def list_manifests(self) -> list[PolicyManifest]:
        if not self.policies_dir.exists():
            return []
        out: list[PolicyManifest] = []
        for man_file in sorted(self.policies_dir.glob("*.manifest.json")):
            out.append(PolicyManifest.from_dict(json.loads(man_file.read_text())))
        return out

    def set_active(self, policy_id: str, reason: str) -> None:
        """Promote a policy to active by flipping the pointer.

        The artifact itself is never modified. The previous active policy
        (if any) is marked archived in its manifest.
        """
        self._ensure_dirs()
        man = self.load_manifest(policy_id)
        if man.promotion_status == PROMOTION_STATUS_REJECTED:
            raise RuntimeError(
                f"cannot activate rejected policy {policy_id}"
            )
        # Archive previous active.
        previous = self.get_active_id()
        if previous is not None and previous != policy_id:
            try:
                prev_man = self.load_manifest(previous)
                prev_man.promotion_status = PROMOTION_STATUS_ARCHIVED
                prev_man.promotion_timestamp = _utc_now_iso()
                prev_man.promotion_reason = f"superseded by {policy_id}"
                self._rewrite_manifest(prev_man)
            except FileNotFoundError:
                pass
        man.promotion_status = PROMOTION_STATUS_ACTIVE
        man.promotion_timestamp = _utc_now_iso()
        man.promotion_reason = reason
        self._rewrite_manifest(man)
        self.active_pointer.write_text(
            json.dumps({"policy_id": policy_id}, sort_keys=True)
        )

    def reject(self, policy_id: str, reason: str) -> None:
        man = self.load_manifest(policy_id)
        man.promotion_status = PROMOTION_STATUS_REJECTED
        man.promotion_timestamp = _utc_now_iso()
        man.promotion_reason = reason
        self._rewrite_manifest(man)

    def set_shadow(self, policy_id: str) -> None:
        man = self.load_manifest(policy_id)
        man.promotion_status = PROMOTION_STATUS_SHADOW
        man.promotion_timestamp = _utc_now_iso()
        self._rewrite_manifest(man)

    def _rewrite_manifest(self, manifest: PolicyManifest) -> None:
        man = self.manifest_path(manifest.policy_id)
        manifest.manifest_path = str(man)
        text = json.dumps(manifest.to_dict(), sort_keys=True)
        manifest.manifest_sha256 = _sha256_text(text)
        man.write_text(text)

    def get_active_id(self) -> str | None:
        if not self.active_pointer.exists():
            return None
        data = json.loads(self.active_pointer.read_text())
        return data.get("policy_id")

    def load_active(self) -> tuple[LearnedPolicy, PolicyManifest] | None:
        pid = self.get_active_id()
        if pid is None:
            return None
        return self.load_policy(pid), self.load_manifest(pid)


def _utc_now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()
