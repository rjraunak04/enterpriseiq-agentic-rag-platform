"""Tests for deterministic and allowlisted agent routing."""

from collections.abc import Mapping

import pytest

from enterpriseiq.agent import (
    AgentDecision,
    AgentQuery,
    AgentRoutingError,
    DecisionType,
    DeterministicAgentRouter,
    RoutingConfig,
    ToolContext,
    ToolDefinition,
    ToolParameter,
    ToolRegistry,
)


def handler(
    arguments: Mapping[str, object],
    context: ToolContext,
) -> object:
    """Simple deterministic handler used only by routing tests."""
    return {
        "arguments": dict(arguments),
        "query_id": context.query_id,
    }


def build_registry() -> ToolRegistry:
    """Build the allowlisted registry expected by the router."""
    return ToolRegistry(
        (
            ToolDefinition(
                name="search_knowledge",
                description="Search enterprise knowledge.",
                handler=handler,
                parameters=(
                    ToolParameter(
                        name="query",
                        value_type=str,
                        description="Enterprise search query.",
                    ),
                    ToolParameter(
                        name="top_k",
                        value_type=int,
                        description="Maximum number of results.",
                        required=False,
                        default=5,
                    ),
                ),
            ),
            ToolDefinition(
                name="get_document",
                description="Retrieve an enterprise document.",
                handler=handler,
                parameters=(
                    ToolParameter(
                        name="document_id",
                        value_type=str,
                        description="Exact document identifier.",
                    ),
                ),
            ),
            ToolDefinition(
                name="list_documents",
                description="List enterprise documents.",
                handler=handler,
                parameters=(
                    ToolParameter(
                        name="department",
                        value_type=str,
                        description="Optional department filter.",
                        required=False,
                        default=None,
                    ),
                ),
            ),
            ToolDefinition(
                name="list_departments",
                description="List enterprise departments.",
                handler=handler,
            ),
        )
    )


@pytest.fixture
def router() -> DeterministicAgentRouter:
    """Return the default deterministic router."""
    return DeterministicAgentRouter(build_registry())


def make_query(
    text: str,
    metadata: Mapping[str, object] | None = None,
) -> AgentQuery:
    """Build an AgentQuery using the canonical CP-01 contract."""
    return AgentQuery(
        text=text,
        metadata=metadata or {},
    )


def test_config_defaults() -> None:
    config = RoutingConfig()

    assert config.default_top_k == 5
    assert config.max_top_k == 20
    assert config.allow_direct_rag is True


@pytest.mark.parametrize(
    ("default_top_k", "max_top_k"),
    [
        (0, 20),
        (True, 20),
        (5, 0),
        (5, True),
        (21, 20),
    ],
)
def test_invalid_config_rejected(
    default_top_k: object,
    max_top_k: object,
) -> None:
    with pytest.raises(AgentRoutingError):
        RoutingConfig(
            default_top_k=default_top_k,  # type: ignore[arg-type]
            max_top_k=max_top_k,  # type: ignore[arg-type]
        )


def test_non_boolean_direct_rag_config_rejected() -> None:
    with pytest.raises(AgentRoutingError):
        RoutingConfig(
            allow_direct_rag="yes",  # type: ignore[arg-type]
        )


def test_router_requires_tool_registry() -> None:
    with pytest.raises(AgentRoutingError):
        DeterministicAgentRouter(
            "registry",  # type: ignore[arg-type]
        )


def test_default_query_routes_to_direct_rag(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("What is the remote work policy?")
    )

    assert decision.decision_type is DecisionType.DIRECT_RAG
    assert decision.tool_call is None


def test_list_departments_routes_to_tool(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("List departments")
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "list_departments"
    assert dict(decision.tool_call.arguments) == {}


def test_show_departments_routes_to_tool(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("Show departments")
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "list_departments"


def test_list_documents_routes_to_tool(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("Show documents")
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "list_documents"
    assert dict(decision.tool_call.arguments) == {
        "department": None,
    }


def test_list_documents_accepts_department(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "List documents",
            {"department": "HR"},
        )
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "list_documents"
    assert dict(decision.tool_call.arguments) == {
        "department": "HR",
    }


def test_exact_document_routes_with_document_id(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Get document",
            {"document_id": "remote-work"},
        )
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "get_document"
    assert dict(decision.tool_call.arguments) == {
        "document_id": "remote-work",
    }


def test_exact_document_without_id_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("Get document")
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_exact_document_with_blank_id_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Get document",
            {"document_id": " "},
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_search_routes_to_search_tool(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Search knowledge about expense approvals"
        )
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "search_knowledge"
    assert dict(decision.tool_call.arguments) == {
        "query": "Search knowledge about expense approvals",
        "top_k": 5,
    }


def test_search_accepts_explicit_search_query(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Search knowledge",
            {
                "search_query": "expense approval",
                "top_k": 3,
            },
        )
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "search_knowledge"
    assert dict(decision.tool_call.arguments) == {
        "query": "expense approval",
        "top_k": 3,
    }


def test_search_strips_explicit_search_query(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Search knowledge",
            {
                "search_query": "  remote work  ",
                "top_k": 2,
            },
        )
    )

    assert decision.tool_call is not None
    assert dict(decision.tool_call.arguments) == {
        "query": "remote work",
        "top_k": 2,
    }


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        21,
        True,
        "five",
    ],
)
def test_invalid_search_top_k_abstains(
    router: DeterministicAgentRouter,
    top_k: object,
) -> None:
    decision = router.route(
        make_query(
            "Search knowledge",
            {"top_k": top_k},
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_blank_explicit_search_query_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Search knowledge",
            {"search_query": " "},
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_explicit_allowlisted_tool_routes(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use the enterprise tool",
            {
                "tool_name": "list_departments",
                "tool_arguments": {},
            },
        )
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "list_departments"


def test_explicit_search_tool_routes_with_arguments(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use the enterprise search tool",
            {
                "tool_name": "search_knowledge",
                "tool_arguments": {
                    "query": "security incidents",
                    "top_k": 4,
                },
            },
        )
    )

    assert decision.decision_type is DecisionType.TOOL_CALL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "search_knowledge"
    assert dict(decision.tool_call.arguments) == {
        "query": "security incidents",
        "top_k": 4,
    }


def test_explicit_unknown_tool_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use unknown tool",
            {
                "tool_name": "unknown_tool",
                "tool_arguments": {},
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_explicit_blank_tool_name_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use tool",
            {
                "tool_name": " ",
                "tool_arguments": {},
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_explicit_tool_arguments_must_be_mapping(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use list departments",
            {
                "tool_name": "list_departments",
                "tool_arguments": "invalid",
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_explicit_tool_invalid_arguments_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use search",
            {
                "tool_name": "search_knowledge",
                "tool_arguments": {},
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_explicit_tool_unknown_argument_abstains(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Use department tool",
            {
                "tool_name": "list_departments",
                "tool_arguments": {
                    "unexpected": "value",
                },
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


@pytest.mark.parametrize(
    "query_text",
    [
        "Run shell command whoami",
        "Execute shell command",
        "Execute Python to read secrets",
        "Run Python code",
        "Delete database",
        "Drop database production",
        "Drop table users",
        "Write file to disk",
        "Delete file secrets.txt",
        "Remove file configuration",
        "Use os.system to execute this",
        "Use subprocess to launch a command",
        "Open powershell",
        "Run cmd.exe",
    ],
)
def test_unsafe_requests_abstain(
    router: DeterministicAgentRouter,
    query_text: str,
) -> None:
    decision = router.route(
        make_query(query_text)
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_unsafe_text_overrides_explicit_safe_tool(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "Run shell but use list_departments",
            {
                "tool_name": "list_departments",
                "tool_arguments": {},
            },
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_missing_required_registry_tool_abstains() -> None:
    router = DeterministicAgentRouter(
        ToolRegistry()
    )

    decision = router.route(
        make_query("List departments")
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_missing_search_tool_abstains() -> None:
    router = DeterministicAgentRouter(
        ToolRegistry()
    )

    decision = router.route(
        make_query(
            "Search knowledge about remote work"
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_direct_rag_can_be_disabled() -> None:
    router = DeterministicAgentRouter(
        build_registry(),
        RoutingConfig(
            allow_direct_rag=False,
        ),
    )

    decision = router.route(
        make_query(
            "What is the remote work policy?"
        )
    )

    assert decision.decision_type is DecisionType.ABSTAIN
    assert decision.tool_call is None


def test_route_is_deterministic(
    router: DeterministicAgentRouter,
) -> None:
    query = make_query(
        "List departments"
    )

    first = router.route(query)
    second = router.route(query)

    assert first == second


def test_search_route_is_deterministic(
    router: DeterministicAgentRouter,
) -> None:
    query = make_query(
        "Search knowledge",
        {
            "search_query": "expense policy",
            "top_k": 5,
        },
    )

    first = router.route(query)
    second = router.route(query)

    assert first == second


def test_decision_is_agent_decision(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("What is the SLA?")
    )

    assert isinstance(
        decision,
        AgentDecision,
    )


def test_direct_rag_decision_contains_reason(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query(
            "What is the travel policy?"
        )
    )

    assert decision.reason
    assert decision.decision_type is DecisionType.DIRECT_RAG


def test_tool_decision_contains_reason(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("List departments")
    )

    assert decision.reason
    assert decision.decision_type is DecisionType.TOOL_CALL


def test_abstention_contains_reason(
    router: DeterministicAgentRouter,
) -> None:
    decision = router.route(
        make_query("Delete database")
    )

    assert decision.reason
    assert decision.decision_type is DecisionType.ABSTAIN