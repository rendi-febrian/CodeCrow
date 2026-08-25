from rag_pipeline.core.review_grouping import (
    review_groups_from_architecture_payloads,
)
from rag_pipeline.api.routers.pr import _review_groups


def test_review_groups_project_only_changed_paths_from_neutral_graph_facts():
    payloads = [{
        "architecture_plugin": "magento",
        "plugin_graph_facts": [
            {
                "kind": "magento-di-effective-preference",
                "path": "app/code/Acme/Checkout/etc/di.xml",
                "related_paths": [
                    "app/code/Acme/Checkout/Api/CartInterface.php",
                    "app/code/Acme/Checkout/Model/Cart.php",
                    "vendor/magento/framework/ObjectManager.php",
                ],
            },
            {
                "kind": "magento-webapi-acl",
                "path": "app/code/Acme/Checkout/etc/webapi.xml",
                "related_paths": [
                    "app/code/Acme/Checkout/etc/acl.xml",
                ],
            },
        ],
    }]
    changed = [
        "app/code/Acme/Checkout/Model/Cart.php",
        "app/code/Acme/Checkout/etc/di.xml",
        "app/code/Acme/Checkout/etc/webapi.xml",
        "app/code/Acme/Checkout/etc/acl.xml",
    ]

    assert review_groups_from_architecture_payloads(payloads, changed) == [
        [
            "app/code/Acme/Checkout/Model/Cart.php",
            "app/code/Acme/Checkout/etc/di.xml",
        ],
        [
            "app/code/Acme/Checkout/etc/acl.xml",
            "app/code/Acme/Checkout/etc/webapi.xml",
        ],
    ]


def test_review_groups_merge_overlapping_facts_deterministically():
    payloads = [
        {
            "plugin_graph_facts": [{
                "path": r"\app\code\Acme\etc\events.xml",
                "related_paths": ["app/code/Acme/Observer/First.php"],
            }],
        },
        {
            "plugin_graph_facts": [{
                "path": "app/code/Acme/Observer/First.php",
                "related_paths": ["app/code/Acme/Observer/Second.php"],
            }],
        },
    ]
    changed = [
        "app/code/Acme/Observer/Second.php",
        "app/code/Acme/etc/events.xml",
        "app/code/Acme/Observer/First.php",
        "unrelated.php",
    ]

    assert review_groups_from_architecture_payloads(payloads, changed) == [[
        "app/code/Acme/Observer/First.php",
        "app/code/Acme/Observer/Second.php",
        "app/code/Acme/etc/events.xml",
    ]]


def test_review_groups_ignore_single_changed_endpoint_and_malformed_facts():
    payloads = [
        {
            "plugin_graph_facts": [
                {
                    "path": "changed.php",
                    "related_paths": ["repository-only.php"],
                },
                "not-a-fact",
            ],
        },
        {"plugin_graph_facts": "not-a-list"},
    ]

    assert review_groups_from_architecture_payloads(
        payloads,
        ["changed.php", "other.php"],
    ) == []


def test_pr_overlay_facts_override_base_relations_for_complete_and_deleted_paths():
    changed = [
        "complete.py",
        "deleted.py",
        "partial.py",
        "other-partial.py",
    ]
    overlay_payloads = [{
        "plugin_graph_facts": [{
            "path": "complete.py",
            "related_paths": ["deleted.py"],
        }],
    }]
    target_payloads = [{
        "plugin_graph_facts": [
            {
                "path": "partial.py",
                "related_paths": ["complete.py"],
            },
            {
                "path": "partial.py",
                "related_paths": ["other-partial.py"],
            },
            {
                "path": "deleted.py",
                "related_paths": ["partial.py"],
            },
        ],
    }]

    assert _review_groups(
        changed,
        overlay_payloads,
        target_payloads,
        ["partial.py", "other-partial.py"],
    ) == [
        ["complete.py", "deleted.py"],
        ["other-partial.py", "partial.py"],
    ]
