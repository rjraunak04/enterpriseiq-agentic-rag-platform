"""Tests for the typed EnterpriseIQ tool registry."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import pytest

from enterpriseiq.agent import (
    ToolContext,
    ToolDefinition,
    ToolInputError,
    ToolNotFoundError,
    ToolParameter,
    ToolRegistrationError,
    ToolRegistry,
)


def echo_handler(
    arguments: Mapping[str, object],
    context: ToolContext,
) -> object:
    return {
        "arguments": dict(arguments),
        "query_id": context.query_id,
    }


def search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_knowledge",
        description="Search enterprise knowledge.",
        handler=echo_handler,
        parameters=(
            ToolParameter(
                name="query",
                value_type=str,
                description="Knowledge search query.",
            ),
            ToolParameter(
                name="top_k",
                value_type=int,
                description="Maximum results.",
                required=False,
                default=5,
            ),
        ),
        metadata={"category": "knowledge"},
    )


def test_parameter_normalization() -> None:
    parameter = ToolParameter(
        name=" query ",
        value_type=str,
        description=" Search query. ",
    )
    assert parameter.name == "query"
    assert parameter.description == "Search query."


def test_parameter_requires_python_type() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolParameter(
            name="query",
            value_type="str",  # type: ignore[arg-type]
            description="Query.",
        )


def test_default_must_match_declared_type() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolParameter(
            name="top_k",
            value_type=int,
            description="Limit.",
            required=False,
            default="5",
        )


def test_bool_rejected_as_integer_default() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolParameter(
            name="top_k",
            value_type=int,
            description="Limit.",
            required=False,
            default=True,
        )


def test_definition_requires_callable_handler() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolDefinition(
            name="search",
            description="Search.",
            handler="invalid",  # type: ignore[arg-type]
        )


def test_definition_rejects_duplicate_parameters() -> None:
    parameter = ToolParameter(
        name="query",
        value_type=str,
        description="Query.",
    )
    with pytest.raises(ToolRegistrationError):
        ToolDefinition(
            name="search",
            description="Search.",
            handler=echo_handler,
            parameters=(parameter, parameter),
        )


def test_definition_is_frozen() -> None:
    tool = search_tool()
    with pytest.raises(FrozenInstanceError):
        tool.name = "changed"  # type: ignore[misc]


def test_definition_metadata_is_immutable() -> None:
    tool = search_tool()
    with pytest.raises(TypeError):
        tool.metadata["category"] = "changed"


def test_argument_validation_applies_default() -> None:
    arguments = search_tool().validate_arguments(
        {"query": "remote work"}
    )
    assert arguments == {
        "query": "remote work",
        "top_k": 5,
    }


def test_validated_arguments_are_immutable() -> None:
    arguments = search_tool().validate_arguments(
        {"query": "remote work"}
    )
    with pytest.raises(TypeError):
        arguments["query"] = "changed"


def test_missing_required_argument_rejected() -> None:
    with pytest.raises(ToolInputError):
        search_tool().validate_arguments({})


def test_unknown_argument_rejected() -> None:
    with pytest.raises(ToolInputError):
        search_tool().validate_arguments(
            {
                "query": "remote work",
                "shell": "whoami",
            }
        )


def test_wrong_argument_type_rejected() -> None:
    with pytest.raises(ToolInputError):
        search_tool().validate_arguments(
            {
                "query": "remote work",
                "top_k": "five",
            }
        )


def test_bool_rejected_for_integer_argument() -> None:
    with pytest.raises(ToolInputError):
        search_tool().validate_arguments(
            {
                "query": "remote work",
                "top_k": True,
            }
        )


def test_float_parameter_accepts_integer() -> None:
    tool = ToolDefinition(
        name="threshold",
        description="Threshold tool.",
        handler=echo_handler,
        parameters=(
            ToolParameter(
                name="score",
                value_type=float,
                description="Score.",
            ),
        ),
    )
    assert tool.validate_arguments({"score": 1})["score"] == 1


def test_registry_register_and_get() -> None:
    tool = search_tool()
    registry = ToolRegistry()
    registry.register(tool)

    assert registry.get("search_knowledge") is tool
    assert registry.contains("search_knowledge")
    assert len(registry) == 1


def test_duplicate_registration_rejected() -> None:
    tool = search_tool()
    registry = ToolRegistry((tool,))

    with pytest.raises(ToolRegistrationError):
        registry.register(tool)


def test_unknown_tool_rejected() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("delete_database")


def test_blank_tool_lookup_rejected() -> None:
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().get(" ")


def test_registry_order_is_deterministic() -> None:
    alpha = ToolDefinition(
        name="alpha",
        description="Alpha tool.",
        handler=echo_handler,
    )
    zebra = ToolDefinition(
        name="zebra",
        description="Zebra tool.",
        handler=echo_handler,
    )

    registry = ToolRegistry((zebra, alpha))

    assert registry.names() == ("alpha", "zebra")
    assert registry.list_tools() == (alpha, zebra)


def test_non_definition_registration_rejected() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolRegistry().register("shell")  # type: ignore[arg-type]


def test_string_registry_input_rejected() -> None:
    with pytest.raises(ToolRegistrationError):
        ToolRegistry("shell")  # type: ignore[arg-type]


def test_context_normalizes_query_id() -> None:
    context = ToolContext(
        query_id=" q-1 ",
        metadata={"tenant": "demo"},
    )
    assert context.query_id == "q-1"


def test_context_metadata_is_immutable() -> None:
    context = ToolContext(metadata={"tenant": "demo"})

    with pytest.raises(TypeError):
        context.metadata["tenant"] = "changed"


def test_context_rejects_blank_query_id() -> None:
    with pytest.raises(ToolInputError):
        ToolContext(query_id=" ")


def test_registered_handler_can_be_called() -> None:
    tool = search_tool()
    arguments = tool.validate_arguments(
        {"query": "remote work"}
    )

    output = tool.handler(
        arguments,
        ToolContext(query_id="q-1"),
    )

    assert output == {
        "arguments": {
            "query": "remote work",
            "top_k": 5,
        },
        "query_id": "q-1",
    }


def test_registry_is_explicit_allowlist() -> None:
    registry = ToolRegistry((search_tool(),))

    assert registry.contains("search_knowledge")
    assert not registry.contains("shell")
    assert not registry.contains("python")
    assert not registry.contains("delete_database")