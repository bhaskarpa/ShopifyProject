import unittest
from dataclasses import dataclass
from typing import List

from MultiTenantDataPrivacyGateway import PermissionLevel, CompiledPathRule, PathAction, register_schema, \
    RedactionContext, DataPrivacyGateway, PrivacyPlanCompiler, TenantPrivacyPolicy, TenantPolicyStore


# from privacy_gateway import (
#     DataPrivacyGateway,
#     PrivacyPlanCompiler,
#     TenantPolicyStore,
#     TenantPrivacyPolicy,
#     RedactionContext,
#     PermissionLevel,
#     CompiledPathRule,
#     PathAction,
#     register_schema,
#     register_privacy_rule,
# )


class DataPrivacyGatewayTests(unittest.TestCase):

    def setUp(self):
        self.tenant_store = TenantPolicyStore()
        self.tenant_store.register_policy(
            TenantPrivacyPolicy(
                tenant_id="tenant_a",
                default_permission=PermissionLevel.ANALYTICS_ONLY,
                enabled_privacy_rules=["gdpr"],
            )
        )

        self.compiler = PrivacyPlanCompiler(self.tenant_store)
        self.gateway = DataPrivacyGateway(self.compiler)

    def context(
        self,
        permission_level=PermissionLevel.ANALYTICS_ONLY,
        schema_name="order",
        privacy_rules=None,
    ):
        return RedactionContext(
            tenant_id="tenant_a",
            schema_name=schema_name,
            privacy_rules=privacy_rules if privacy_rules is not None else ["gdpr"],
            permission_level=permission_level,
            requester_id="analyst-1",
            purpose="analytics",
        )

    def order_payload(self):
        return {
            "order_id": "o123",
            "customer": {
                "name": "Jane Doe",
                "email": "jane@example.com",
                "phone": "555-111-2222",
                "zip_code": "95110",
            },
            "shipping_address": {
                "line1": "123 Main St",
                "zip_code": "95110",
            },
            "billing_address": {
                "line1": "999 Market St",
            },
            "amount": 120.50,
        }

    # =========================================================
    # Functional & Policy Tests
    # =========================================================

    def test_analytics_only_redacts_customer_name_but_keeps_zip_code(self):
        result = self.gateway.redact(
            self.order_payload(),
            self.context(PermissionLevel.ANALYTICS_ONLY),
        )

        self.assertNotIn("name", result["customer"])
        self.assertEqual(result["customer"]["zip_code"], "95110")

    def test_analytics_only_nullifies_email(self):
        result = self.gateway.redact(
            self.order_payload(),
            self.context(PermissionLevel.ANALYTICS_ONLY),
        )

        self.assertIsNone(result["customer"]["email"])

    def test_masked_permission_masks_email(self):
        result = self.gateway.redact(
            self.order_payload(),
            self.context(PermissionLevel.MASKED),
        )

        self.assertEqual(result["customer"]["email"], "j***@example.com")

    def test_masked_permission_removes_addresses_and_nullifies_phone(self):
        result = self.gateway.redact(
            self.order_payload(),
            self.context(PermissionLevel.MASKED),
        )

        self.assertNotIn("shipping_address", result)
        self.assertNotIn("billing_address", result)
        self.assertIsNone(result["customer"]["phone"])

    def test_full_access_keeps_schema_fields(self):
        result = self.gateway.redact(
            self.order_payload(),
            self.context(PermissionLevel.FULL_ACCESS),
        )

        self.assertEqual(result["customer"]["name"], "Jane Doe")
        self.assertEqual(result["customer"]["email"], "jane@example.com")
        self.assertIn("shipping_address", result)

    def test_applies_gdpr_policy_rule(self):
        payload = self.order_payload()
        payload["consent"] = {
            "ip_address": "10.0.0.1",
        }

        result = self.gateway.redact(
            payload,
            self.context(
                permission_level=PermissionLevel.ANALYTICS_ONLY,
                privacy_rules=["gdpr"],
            ),
        )

        self.assertIsNone(result["consent"]["ip_address"])
        self.assertNotIn("name", result["customer"])

    def test_applies_ccpa_policy_rule(self):
        payload = self.order_payload()
        payload["device"] = {
            "ip_address": "10.0.0.1",
        }

        result = self.gateway.redact(
            payload,
            self.context(
                permission_level=PermissionLevel.ANALYTICS_ONLY,
                privacy_rules=["ccpa"],
            ),
        )

        self.assertNotIn("phone", result["customer"])
        self.assertIsNone(result["device"]["ip_address"])

    def test_uses_tenant_default_policy_when_context_rules_empty(self):
        result = self.gateway.redact(
            self.order_payload(),
            self.context(
                permission_level=PermissionLevel.ANALYTICS_ONLY,
                privacy_rules=[],
            ),
        )

        self.assertNotIn("name", result["customer"])
        self.assertIsNone(result["customer"]["email"])

    # =========================================================
    # Schema Evolution Tests
    # =========================================================

    def test_can_add_new_schema_to_registry(self):
        schema_name = f"invoice_test_{id(self)}"

        @register_schema(schema_name)
        @dataclass(frozen=True)
        class InvoiceSchema:
            name: str = schema_name

            def pii_path_rules(self, context) -> List[CompiledPathRule]:
                if context.permission_level == PermissionLevel.ANALYTICS_ONLY:
                    return [
                        CompiledPathRule(("billing", "email"), PathAction.NULLIFY),
                        CompiledPathRule(("billing", "name"), PathAction.REMOVE),
                    ]
                return []

        payload = {
            "invoice_id": "i123",
            "billing": {
                "name": "John Doe",
                "email": "john@example.com",
                "zip_code": "94016",
            },
        }

        result = self.gateway.redact(
            payload,
            self.context(
                schema_name=schema_name,
                permission_level=PermissionLevel.ANALYTICS_ONLY,
            ),
        )

        self.assertNotIn("name", result["billing"])
        self.assertIsNone(result["billing"]["email"])
        self.assertEqual(result["billing"]["zip_code"], "94016")

    def test_unknown_field_not_in_schema_is_preserved_by_current_implementation(self):
        """
        Current implementation only applies compiled path rules.
        It does not default-deny unknown fields.
        """
        payload = self.order_payload()
        payload["customer"]["unexpected_field"] = "some-value"

        result = self.gateway.redact(
            payload,
            self.context(PermissionLevel.ANALYTICS_ONLY),
        )

        self.assertIn("unexpected_field", result["customer"])
        self.assertEqual(result["customer"]["unexpected_field"], "some-value")

    def test_policy_mismatch_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gateway.redact(
                self.order_payload(),
                self.context(
                    permission_level=PermissionLevel.ANALYTICS_ONLY,
                    privacy_rules=["missing_policy"],
                ),
            )

    def test_unknown_schema_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.gateway.redact(
                self.order_payload(),
                self.context(
                    schema_name="missing_schema",
                    permission_level=PermissionLevel.ANALYTICS_ONLY,
                ),
            )

    # =========================================================
    # Cache Tests
    # =========================================================

    def test_compiler_reuses_cached_plan(self):
        context = self.context(PermissionLevel.ANALYTICS_ONLY)

        plan1 = self.compiler.compile(context)
        plan2 = self.compiler.compile(context)

        self.assertIs(plan1, plan2)
        self.assertEqual(len(self.compiler.cache), 1)

    # =========================================================
    # Security Behavior Tests for Current Implementation
    # =========================================================

    def test_sensitive_field_not_in_compiled_rules_is_not_redacted_in_current_implementation(self):
        """
        This documents current behavior.

        To make this pass as a security redaction test, add explicit schema or
        global deny rules for ssn and credit_card_number.
        """
        payload = self.order_payload()
        payload["customer"]["ssn"] = "123-45-6789"
        payload["customer"]["credit_card_number"] = "4111111111111111"

        result = self.gateway.redact(
            payload,
            self.context(PermissionLevel.FULL_ACCESS),
        )

        self.assertEqual(result["customer"]["ssn"], "123-45-6789")
        self.assertEqual(
            result["customer"]["credit_card_number"],
            "4111111111111111",
        )

    def test_malicious_obfuscated_email_key_is_not_redacted_by_path_based_engine(self):
        """
        Current fast-path engine does not scan keys or values.
        It only applies registered compiled paths.
        """
        payload = {
            "customer": {
                "e_m_a_i_l": "attacker@example.com",
                "zip_code": "95110",
            }
        }

        result = self.gateway.redact(
            payload,
            self.context(PermissionLevel.ANALYTICS_ONLY),
        )

        self.assertEqual(result["customer"]["e_m_a_i_l"], "attacker@example.com")

    def test_uppercase_email_key_is_not_redacted_unless_schema_contains_that_path(self):
        payload = {
            "customer": {
                "EMAIL": "upper@example.com",
                "zip_code": "95110",
            }
        }

        result = self.gateway.redact(
            payload,
            self.context(PermissionLevel.ANALYTICS_ONLY),
        )

        self.assertEqual(result["customer"]["EMAIL"], "upper@example.com")


if __name__ == "__main__":
    unittest.main()