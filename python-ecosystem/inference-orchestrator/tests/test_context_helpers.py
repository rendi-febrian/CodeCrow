"""Unit tests for deterministic repository-context formatting."""
import pytest
from service.review.orchestrator.context_helpers import (
    format_rag_context,
    rag_evidence_id,
)


SAMPLE_DIFF = """\
diff --git a/src/OrderService.java b/src/OrderService.java
--- a/src/OrderService.java
+++ b/src/OrderService.java
@@ -10,6 +10,10 @@
+    public Order createOrder(CreateOrderRequest request) {
+        OrderValidator validator = new OrderValidator();
+        validator.validate(request);
+        return orderRepository.save(request.toOrder());
+    }
-    public void oldMethod() {
"""


# ── extract_symbols_from_diff ────────────────────────────────────



# ── extract_diff_snippets ────────────────────────────────────────



# ── get_diff_snippets_for_batch ──────────────────────────────────



# ── format_rag_context ───────────────────────────────────────────

class TestFormatRagContext:

    def test_empty_input(self):
        assert format_rag_context(None) == ""
        assert format_rag_context({}) == ""
        assert format_rag_context({"relevant_code": []}) == ""

    def test_basic_chunk(self):
        rag = {
            "relevant_code": [
                {
                    "text": "def process(): pass",
                    "metadata": {"path": "src/proc.py", "content_type": "functions_classes"},
                    "_match_type": "definition",
                }
            ]
        }
        result = format_rag_context(rag)
        assert "src/proc.py" in result
        assert "def process(): pass" in result

    def test_retrieved_chunk_has_stable_citation_id(self):
        chunk = {
            "text": "exact relation",
            "metadata": {
                "path": "__analysis_architecture__/relation.context",
                "architecture_key": "relation-key",
            },
            "_match_type": "architecture_relation",
        }

        first = rag_evidence_id(chunk)
        second = rag_evidence_id(dict(chunk))
        result = format_rag_context({"relevant_code": [chunk]})

        assert first == second
        assert first.startswith("RAG-")
        assert f"Evidence ID: {first}" in result

    def test_prompt_visible_exact_chunk_is_available_as_citation(self):
        chunk = {
            "text": "def process(): pass",
            "metadata": {"path": "src/proc.py"},
            "_match_type": "definition",
        }
        visible = {}

        result = format_rag_context(
            {"relevant_code": [chunk]},
            visible_evidence_by_id=visible,
        )

        evidence_id = rag_evidence_id(chunk)
        assert f"Evidence ID: {evidence_id}" in result
        assert visible == {evidence_id: ()}

    def test_prompt_visible_graph_fact_is_available_to_validation(self):
        fact = {
            "kind": "java-type",
            "source": "com.example.App",
            "relation": "declares",
            "target": "App",
            "path": "src/App.java",
            "line": 1,
        }
        chunk = {
            "text": (
                "[java-type] com.example.App declares App\n"
                "public class App {}"
            ),
            "metadata": {
                "path": "src/App.java",
                "plugin_graph_facts": [fact],
            },
            "_match_type": "architecture_relation",
        }
        visible = {}

        result = format_rag_context(
            {"relevant_code": [chunk]},
            visible_evidence_by_id=visible,
        )

        assert "[java-type]" in result
        assert visible == {rag_evidence_id(chunk): (fact,)}

    def test_metadata_facts_are_rendered_once_without_mutating_stored_source(self):
        fact = {
            "kind": "python-call",
            "source": "service.review",
            "relation": "calls",
            "target": "validate",
            "path": "src/service.py",
            "line": 10,
        }
        fact_line = "[python-call] service.review calls validate"
        first = {
            "text": "def review():\n    validate()",
            "metadata": {
                "path": "src/service.py",
                "plugin_graph_facts": [fact],
            },
            "_match_type": "changed_file",
            "_source": "deterministic",
        }
        second = {
            "text": "def validate():\n    return True",
            "metadata": {
                "path": "src/service.py",
                "plugin_graph_facts": [fact],
            },
            "_match_type": "definition",
            "_source": "deterministic",
        }
        visible = {}

        result = format_rag_context(
            {"relevant_code": [first, second]},
            visible_evidence_by_id=visible,
        )

        assert result.count(fact_line) == 1
        assert "def review()" in result
        assert "def validate()" in result
        assert visible == {
            rag_evidence_id(first): (),
            rag_evidence_id(second): (fact,),
        }

    def test_metadata_fact_hidden_by_chunk_truncation_cannot_validate(self):
        visible_fact = {
            "kind": "magento-effective-route",
            "source": "checkout",
            "relation": "handled-by-module",
            "target": "Acme_Checkout",
            "path": "app/code/Acme/Checkout/etc/frontend/routes.xml",
            "line": 1,
        }
        hidden_fact = {
            "kind": "magento-webapi-route",
            "source": "POST /V1/cart",
            "relation": "invokes",
            "target": "Acme\\Api\\CartInterface::save",
            "path": "app/code/Acme/Checkout/etc/webapi.xml",
            "line": 1,
        }
        chunk = {
            "text": (
                "[magento-effective-route] checkout handled-by-module "
                "Acme_Checkout\n"
                + ("x" * 2_000)
                + "\n[magento-webapi-route] POST /V1/cart invokes "
                "Acme\\Api\\CartInterface::save"
            ),
            "metadata": {
                "path": "__analysis_architecture__/magento/routes.context",
                "plugin_graph_facts": [visible_fact, hidden_fact],
            },
            "_match_type": "architecture_relation",
        }
        visible = {}

        format_rag_context(
            {"relevant_code": [chunk]},
            visible_evidence_by_id=visible,
        )

        assert visible[rag_evidence_id(chunk)] == (visible_fact, hidden_fact)

    def test_filters_deleted_files(self):
        rag = {
            "relevant_code": [
                {
                    "text": "old code",
                    "metadata": {"path": "deleted.py"},
                    "_match_type": "definition",
                },
                {
                    "text": "kept code",
                    "metadata": {"path": "kept.py"},
                    "_match_type": "definition",
                },
            ]
        }
        result = format_rag_context(rag, deleted_files=["deleted.py"])
        assert "deleted.py" not in result
        assert "kept.py" in result

    def test_tiered_budgeting(self):
        """Tier 1 (definition) chunks should appear in output."""
        chunks = [
            {
                "text": f"class Base{i}: pass",
                "metadata": {"path": f"src/base{i}.py", "content_type": "functions_classes"},
                "_match_type": "definition",
                "_source": "deterministic",
            }
            for i in range(12)
        ]
        rag = {"relevant_code": chunks}
        result = format_rag_context(rag)
        count = sum(1 for i in range(12) if f"src/base{i}.py" in result)
        assert count == 12

    def test_untyped_context_is_not_rendered(self):
        result = format_rag_context({
            "relevant_code": [{
                "text": "unrelated sibling implementation",
                "metadata": {"path": "src/same_package/sibling.py"},
                "_match_type": "unproved_context",
                "_source": "deterministic",
            }],
        })

        assert result == ""

    def test_focused_architecture_relations_are_not_cut_at_eight(self):
        rag = {
            "relevant_code": [
                {
                    "text": f"[graph-fact] Source{index} resolves-to Target{index}",
                    "metadata": {
                        "path": f"__analysis_architecture__/packet-{index}.context",
                        "architecture_kind": f"kind-{index % 4}",
                    },
                    "_match_type": "architecture_relation",
                    "_source": "pr_indexed",
                }
                for index in range(20)
            ],
        }

        result = format_rag_context(rag)

        assert sum(
            f"Source{index} resolves-to Target{index}" in result
            for index in range(20)
        ) == 20
        assert len(result) <= 32_000

    def test_exact_architecture_relations_are_not_silently_cut_at_sixty_four(self):
        rag = {
            "relevant_code": [
                {
                    "text": (
                        f"[graph-fact] Source{index} resolves-to Target{index}"
                    ),
                    "metadata": {
                        "path": (
                            "__analysis_architecture__/"
                            f"packet-{index}.context"
                        ),
                        "architecture_kind": f"kind-{index % 5}",
                    },
                    "_match_type": "architecture_relation",
                    "_source": "pr_indexed",
                }
                for index in range(65)
            ],
        }

        result = format_rag_context(rag)

        assert "Source64 resolves-to Target64" in result
        assert sum(
            f"Source{index} resolves-to Target{index}" in result
            for index in range(65)
        ) == 65

    def test_complete_structural_context_is_preserved(self):
        rag = {
            "relevant_code": [
                {
                    "text": "class RequiredBase:\n" + ("x = 1\n" * 500),
                    "metadata": {"path": "src/RequiredBase.py"},
                    "_match_type": "definition",
                    "_source": "deterministic",
                },
                {
                    "text": "def merely_similar():\n" + ("return 1\n" * 500),
                    "metadata": {"path": "src/Similar.py"},
                },
            ]
        }

        result = format_rag_context(rag)

        assert "src/RequiredBase.py" in result
        assert "src/Similar.py" not in result
        assert result.count("x = 1") == 500
        assert "Context chunk truncated" not in result
        assert result.count("```") == 2

    def test_multiple_large_structural_chunks_are_bounded(self):
        rag = {
            "relevant_code": [
                {
                    "text": f"class Base{index}:\n" + (f"value_{index} = 1\n" * 600),
                    "metadata": {"path": f"src/Base{index}.py"},
                    "_match_type": "definition",
                    "_source": "deterministic",
                }
                for index in range(8)
            ]
        }

        result = format_rag_context(rag)

        assert "src/Base0.py" in result
        assert "src/Base7.py" not in result
        assert "Context chunk truncated by deterministic prompt budget" in result
        assert len(result) <= 32_000
        assert result.count("```") % 2 == 0

    def test_complete_current_file_chunk_is_removed_but_related_file_remains(self):
        rag = {
            "relevant_code": [
                {
                    "text": "class Reviewed:\n    pass",
                    "metadata": {"path": "src/Reviewed.py"},
                    "_match_type": "changed_file",
                    "_source": "pr_indexed",
                },
                {
                    "text": "class Dependency:\n    pass",
                    "metadata": {"path": "src/Dependency.py"},
                    "_match_type": "definition",
                    "_source": "deterministic",
                },
            ]
        }

        result = format_rag_context(
            rag,
            current_file_complete_paths={"src/Reviewed.py"},
        )

        assert "src/Reviewed.py" not in result
        assert "src/Dependency.py" in result

    def test_complete_current_file_keeps_exact_architecture_evidence(self):
        fact = {
            "kind": "python-call",
            "source": "src.reviewed",
            "relation": "calls",
            "target": "dependency",
            "path": "src/Reviewed.py",
            "related_paths": ["src/Dependency.py"],
            "attributes": {},
        }
        rag = {
            "relevant_code": [
                {
                    "text": (
                        "[python-call] src.reviewed calls dependency\n"
                        "implementation details"
                    ),
                    "metadata": {
                        "path": "src/Reviewed.py",
                        "architecture_key": "python-file:src/Reviewed.py",
                        "plugin_graph_facts": [fact],
                    },
                    "_match_type": "architecture_relation",
                    "_source": "pr_indexed",
                },
            ]
        }
        visible = {}

        result = format_rag_context(
            rag,
            current_file_complete_paths={"src/Reviewed.py"},
            visible_evidence_by_id=visible,
        )

        assert "src/Reviewed.py" in result
        assert "[python-call] src.reviewed calls dependency" in result
        assert tuple(visible.values()) == ((fact,),)

    def test_truncated_current_file_chunk_is_retained(self):
        rag = {
            "relevant_code": [
                {
                    "text": "middle_of_large_file()",
                    "metadata": {"path": "src/Large.py"},
                    "_match_type": "changed_file",
                    "_source": "pr_indexed",
                },
            ]
        }

        result = format_rag_context(
            rag,
            current_file_complete_paths=set(),
        )

        assert "src/Large.py" in result
        assert "middle_of_large_file()" in result

    def test_untyped_documentation_chunk_is_not_structural_context(self):
        rag = {
            "relevant_code": [
                {
                    "text": "readme content",
                    "metadata": {"path": "README.md", "content_type": "documentation"},
                },
            ]
        }
        result = format_rag_context(rag)
        assert result == ""

    def test_exact_evidence_deduplication(self):
        """Repeated retrieval of the same full evidence identity is deduplicated."""
        rag = {
            "relevant_code": [
                {
                    "text": "same content here",
                    "metadata": {
                        "path": "src/a/util.py",
                        "start_line": 4,
                        "end_line": 4,
                    },
                    "_match_type": "definition",
                },
                {
                    "text": "same content here",
                    "metadata": {
                        "path": "src/a/util.py",
                        "start_line": 4,
                        "end_line": 4,
                    },
                    "_match_type": "definition",
                },
            ]
        }
        result = format_rag_context(rag)
        assert result.count("same content here") == 1

    def test_unlocated_exact_duplicate_is_collapsed_by_path_and_text(self):
        chunk = {
            "text": "legacy equal text",
            "metadata": {"path": "src/legacy.py"},
            "_match_type": "definition",
        }

        result = format_rag_context({
            "relevant_code": [chunk, dict(chunk)],
        })

        assert result.count("legacy equal text") == 1

    def test_repeated_point_id_is_deduplicated_before_range_fallback(self):
        text = "def приветствие(): return '👋'"
        rag = {
            "relevant_code": [
                {
                    "id": "qdrant-point-α",
                    "text": text,
                    "metadata": {
                        "path": "src/привет.py",
                        "start_line": 10,
                        "end_line": 10,
                    },
                    "_match_type": "definition",
                },
                {
                    "id": "qdrant-point-α",
                    "text": text,
                    "metadata": {
                        "path": "src/привет.py",
                        "start_line": 30,
                        "end_line": 30,
                    },
                    "_match_type": "definition",
                },
            ]
        }

        result = format_rag_context(rag)

        assert result.count(text) == 1

    @pytest.mark.parametrize(
        ("first_locator", "second_locator"),
        [
            (
                {"start_line": 10, "end_line": 10},
                {"start_line": 30, "end_line": 30},
            ),
            ({"chunk_index": 0}, {"chunk_index": 1}),
        ],
    )
    def test_equal_unicode_text_at_distinct_occurrences_is_not_deduplicated(
        self,
        first_locator,
        second_locator,
    ):
        text = "value = 'однаковий текст 🐦'"
        first = {
            "text": text,
            "metadata": {"path": "src/птах.py", **first_locator},
            "_match_type": "definition",
        }
        second = {
            "text": text,
            "metadata": {"path": "src/птах.py", **second_locator},
            "_match_type": "definition",
        }

        result = format_rag_context({"relevant_code": [first, second]})

        assert result.count(text) == 2
        assert rag_evidence_id(first) != rag_evidence_id(second)
        assert f"Evidence ID: {rag_evidence_id(first)}" in result
        assert f"Evidence ID: {rag_evidence_id(second)}" in result

    def test_same_basename_and_content_in_distinct_paths_are_not_deduplicated(self):
        rag = {
            "relevant_code": [
                {
                    "text": "same content here",
                    "metadata": {"path": "src/a/util.py"},
                    "_match_type": "definition",
                },
                {
                    "text": "same content here",
                    "metadata": {"path": "src/b/util.py"},
                    "_match_type": "definition",
                },
            ]
        }

        result = format_rag_context(rag)

        assert "src/a/util.py" in result
        assert "src/b/util.py" in result
        assert result.count("same content here") == 2

    def test_magento_module_di_files_with_identical_prefixes_remain_distinct(self):
        shared_prefix = "<config>" + (" " * 350)
        rag = {
            "relevant_code": [
                {
                    "text": shared_prefix + "<preference for='Cart' type='CartImpl'/></config>",
                    "metadata": {
                        "path": "app/code/Acme/Cart/etc/di.xml",
                        "architecture_key": "Acme_Cart:global",
                    },
                    "_match_type": "architecture_relation",
                },
                {
                    "text": shared_prefix + "<type name='Checkout'><plugin name='tax'/></type></config>",
                    "metadata": {
                        "path": "app/code/Acme/Checkout/etc/di.xml",
                        "architecture_key": "Acme_Checkout:global",
                    },
                    "_match_type": "architecture_relation",
                },
            ]
        }

        result = format_rag_context(rag)

        assert "app/code/Acme/Cart/etc/di.xml" in result
        assert "app/code/Acme/Checkout/etc/di.xml" in result
        assert "CartImpl" in result
        assert "name='tax'" in result

    def test_stale_base_chunk_from_modified_file(self):
        rag = {
            "relevant_code": [
                {
                    "text": "stale code",
                    "metadata": {"path": "modified.py"},
                },
            ]
        }
        result = format_rag_context(rag, pr_changed_files=["modified.py"])
        assert result == ""

    def test_pr_indexed_not_filtered(self):
        """PR-indexed chunks from modified files should NOT be filtered."""
        rag = {
            "relevant_code": [
                {
                    "text": "fresh indexed code",
                    "metadata": {"path": "modified.py"},
                    "_source": "pr_indexed",
                    "_match_type": "changed_file",
                },
            ]
        }
        result = format_rag_context(rag, pr_changed_files=["modified.py"])
        assert "fresh indexed code" in result

    def test_base_architecture_packet_touching_modified_source_is_filtered(self):
        rag = {
            "relevant_code": [{
                "text": "old effective DI relation",
                "metadata": {
                    "path": "__analysis_architecture__/magento/packet.context",
                    "architecture_context": True,
                    "architecture_paths": [
                        "app/code/Acme/Checkout/etc/di.xml",
                        "app/code/Acme/Checkout/Plugin/OldPlugin.php",
                    ],
                },
                "_source": "deterministic",
                "_match_type": "architecture_relation",
            }]
        }

        result = format_rag_context(
            rag,
            pr_changed_files=["app/code/Acme/Checkout/etc/di.xml"],
        )

        assert result == ""

    def test_pr_architecture_packet_touching_modified_source_is_retained(self):
        rag = {
            "relevant_code": [{
                "text": "new DI relation",
                "metadata": {
                    "path": "__analysis_architecture__/magento/packet.context",
                    "architecture_context": True,
                    "architecture_paths": ["app/code/Acme/Checkout/etc/di.xml"],
                },
                "_source": "pr_indexed",
                "_match_type": "architecture_relation",
            }]
        }

        result = format_rag_context(
            rag,
            pr_changed_files=["app/code/Acme/Checkout/etc/di.xml"],
        )

        assert "new DI relation" in result
