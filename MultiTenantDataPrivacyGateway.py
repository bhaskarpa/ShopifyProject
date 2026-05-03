from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Protocol, Optional, Callable
import copy


# ============================================================
# Permission / Context
# ============================================================

class PermissionLevel(str, Enum):
    FULL_ACCESS = "full_access"
    MASKED = "masked"
    ANALYTICS_ONLY = "analytics_only"


@dataclass(frozen=True)
class RedactionContext:
    tenant_id: str
    schema_name: str
    privacy_rules: List[str]
    permission_level: PermissionLevel
    requester_id: Optional[str] = None
    purpose: Optional[str] = None


# ============================================================
# Fast Path Model
# ============================================================

class PathAction(str, Enum):
    REMOVE = "remove"
    MASK_EMAIL = "mask_email"
    NULLIFY = "nullify"


@dataclass(frozen=True)
class CompiledPathRule:
    """
    Precompiled path rule.

    Example:
      path=("customer", "email")
      action=MASK_EMAIL
    """
    path: tuple[str, ...]
    action: PathAction


@dataclass(frozen=True)
class CompiledPrivacyPlan:
    """
    Built once per tenant + schema + permission level.

    Runtime hot path only applies these compiled rules.
    """
    tenant_id: str
    schema_name: str
    permission_level: PermissionLevel
    rules: List[CompiledPathRule]


# ============================================================
# Registry Infrastructure
# ============================================================

class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Any] = {}

    def register(self, name: str, item: Any) -> None:
        if name in self._items:
            raise ValueError(f"Duplicate registry entry: {name}")
        self._items[name] = item

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise ValueError(f"No registered item named: {name}")
        return self._items[name]


schema_registry = Registry()
privacy_rule_registry = Registry()


def register_schema(name: str) -> Callable:
    def decorator(cls):
        schema_registry.register(name, cls())
        return cls
    return decorator


def register_privacy_rule(name: str) -> Callable:
    def decorator(cls):
        privacy_rule_registry.register(name, cls())
        return cls
    return decorator


# ============================================================
# Interfaces
# ============================================================

class DataSchema(Protocol):
    name: str

    def pii_path_rules(self, context: RedactionContext) -> List[CompiledPathRule]:
        ...


class PrivacyRule(Protocol):
    name: str

    def path_rules(self, context: RedactionContext) -> List[CompiledPathRule]:
        ...


# ============================================================
# Example Schemas
# ============================================================

@register_schema("order")
@dataclass(frozen=True)
class OrderSchema:
    name: str = "order"

    def pii_path_rules(self, context: RedactionContext) -> List[CompiledPathRule]:
        if context.permission_level == PermissionLevel.FULL_ACCESS:
            return []

        if context.permission_level == PermissionLevel.MASKED:
            return [
                CompiledPathRule(("customer", "email"), PathAction.MASK_EMAIL),
                CompiledPathRule(("customer", "phone"), PathAction.NULLIFY),
                CompiledPathRule(("shipping_address",), PathAction.REMOVE),
                CompiledPathRule(("billing_address",), PathAction.REMOVE),
            ]

        if context.permission_level == PermissionLevel.ANALYTICS_ONLY:
            return [
                CompiledPathRule(("customer", "email"), PathAction.NULLIFY),
                CompiledPathRule(("customer", "phone"), PathAction.NULLIFY),
                CompiledPathRule(("customer", "name"), PathAction.REMOVE),
                CompiledPathRule(("shipping_address",), PathAction.REMOVE),
                CompiledPathRule(("billing_address",), PathAction.REMOVE),
            ]

        return []


@register_schema("telemetry")
@dataclass(frozen=True)
class TelemetrySchema:
    name: str = "telemetry"

    def pii_path_rules(self, context: RedactionContext) -> List[CompiledPathRule]:
        if context.permission_level == PermissionLevel.FULL_ACCESS:
            return []

        return [
            CompiledPathRule(("device", "ip_address"), PathAction.NULLIFY),
            CompiledPathRule(("device", "ad_id"), PathAction.NULLIFY),
            CompiledPathRule(("user", "email"), PathAction.MASK_EMAIL),
        ]


# ============================================================
# Example Privacy Rules
# ============================================================

@register_privacy_rule("gdpr")
@dataclass(frozen=True)
class GDPRRule:
    name: str = "gdpr"

    def path_rules(self, context: RedactionContext) -> List[CompiledPathRule]:
        if context.permission_level == PermissionLevel.FULL_ACCESS:
            return []

        return [
            CompiledPathRule(("consent", "ip_address"), PathAction.NULLIFY),
            CompiledPathRule(("customer", "name"), PathAction.REMOVE),
        ]


@register_privacy_rule("ccpa")
@dataclass(frozen=True)
class CCPARule:
    name: str = "ccpa"

    def path_rules(self, context: RedactionContext) -> List[CompiledPathRule]:
        if context.permission_level == PermissionLevel.ANALYTICS_ONLY:
            return [
                CompiledPathRule(("customer", "phone"), PathAction.REMOVE),
                CompiledPathRule(("device", "ip_address"), PathAction.NULLIFY),
            ]

        return []


# ============================================================
# Tenant Policy
# ============================================================

@dataclass(frozen=True)
class TenantPrivacyPolicy:
    tenant_id: str
    default_permission: PermissionLevel
    enabled_privacy_rules: List[str]


class TenantPolicyStore:
    def __init__(self) -> None:
        self.policies: Dict[str, TenantPrivacyPolicy] = {}

    def register_policy(self, policy: TenantPrivacyPolicy) -> None:
        self.policies[policy.tenant_id] = policy

    def get_policy(self, tenant_id: str) -> TenantPrivacyPolicy:
        if tenant_id not in self.policies:
            raise ValueError(f"No policy configured for tenant={tenant_id}")
        return self.policies[tenant_id]


# ============================================================
# Privacy Plan Compiler
# ============================================================

class PrivacyPlanCompiler:
    """
    Compiles tenant + schema + permission into direct path operations.

    This is where we pay the cost of combining:
      - schema rules
      - GDPR/CCPA rules
      - tenant policy

    The runtime engine should reuse cached CompiledPrivacyPlan objects.
    """

    def __init__(self, tenant_store: TenantPolicyStore) -> None:
        self.tenant_store = tenant_store
        self.cache: Dict[tuple[str, str, PermissionLevel, tuple[str, ...]], CompiledPrivacyPlan] = {}

    def compile(self, context: RedactionContext) -> CompiledPrivacyPlan:
        tenant_policy = self.tenant_store.get_policy(context.tenant_id)

        active_rules = (
            context.privacy_rules
            if context.privacy_rules
            else tenant_policy.enabled_privacy_rules
        )

        cache_key = (
            context.tenant_id,
            context.schema_name,
            context.permission_level,
            tuple(sorted(active_rules)),
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        schema: DataSchema = schema_registry.get(context.schema_name)

        rules: List[CompiledPathRule] = []
        rules.extend(schema.pii_path_rules(context))

        for rule_name in active_rules:
            privacy_rule: PrivacyRule = privacy_rule_registry.get(rule_name)
            rules.extend(privacy_rule.path_rules(context))

        plan = CompiledPrivacyPlan(
            tenant_id=context.tenant_id,
            schema_name=context.schema_name,
            permission_level=context.permission_level,
            rules=rules,
        )

        self.cache[cache_key] = plan
        return plan


# ============================================================
# Fast Gateway Engine
# ============================================================

class DataPrivacyGateway:
    """
    High-throughput privacy gateway.

    Runtime behavior:
      - no recursive schema discovery
      - no regex scanning over entire JSON
      - apply direct path operations from a precompiled plan
    """

    def __init__(self, compiler: PrivacyPlanCompiler) -> None:
        self.compiler = compiler

    def redact(
        self,
        payload: Dict[str, Any],
        context: RedactionContext,
    ) -> Dict[str, Any]:
        plan = self.compiler.compile(context)

        result = copy.deepcopy(payload)

        for rule in plan.rules:
            self._apply_rule(result, rule)

        return result

    def _apply_rule(self, payload: Dict[str, Any], rule: CompiledPathRule) -> None:
        parent = self._get_parent(payload, rule.path)
        if parent is None:
            return

        leaf = rule.path[-1]

        if leaf not in parent:
            return

        if rule.action == PathAction.REMOVE:
            parent.pop(leaf, None)

        elif rule.action == PathAction.NULLIFY:
            parent[leaf] = None

        elif rule.action == PathAction.MASK_EMAIL:
            parent[leaf] = self._mask_email(parent[leaf])

    def _get_parent(
        self,
        payload: Dict[str, Any],
        path: tuple[str, ...],
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the parent dict for a path.

        Example:
          path=("customer", "email")
          returns payload["customer"]
        """
        if not path:
            return None

        current: Any = payload

        for part in path[:-1]:
            if not isinstance(current, dict):
                return None
            current = current.get(part)

        return current if isinstance(current, dict) else None

    def _mask_email(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        local, sep, domain = value.partition("@")
        if not sep or not local or not domain:
            return "***"

        return f"{local[0]}***@{domain}"

if __name__ == "__main__":
    tenant_store = TenantPolicyStore()

    tenant_store.register_policy(
        TenantPrivacyPolicy(
            tenant_id="tenant_a",
            default_permission=PermissionLevel.MASKED,
            enabled_privacy_rules=["gdpr"],
        )
    )

    compiler = PrivacyPlanCompiler(tenant_store)
    gateway = DataPrivacyGateway(compiler)

    payload = {
        "order_id": "o123",
        "customer": {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "555-111-2222",
        },
        "shipping_address": {
            "line1": "123 Main St",
            "city": "San Jose",
        },
        "amount": 120.50,
    }

    context = RedactionContext(
        tenant_id="tenant_a",
        schema_name="order",
        privacy_rules=["gdpr"],
        permission_level=PermissionLevel.MASKED,
    )

    redacted = gateway.redact(payload, context)

    print(redacted)