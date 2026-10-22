"""End-to-end agentic RAG orchestration for EnterpriseIQ."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from enterpriseiq.agent.execution import SafeToolExecutor
from enterpriseiq.agent.models import (
    AgentDecision,
    AgentQuery,
    AgentResponse,
    AgentStatus,
    AgentStep,
    AgentTrace,
    DecisionType,
    ToolExecutionStatus,
    ToolResult,
)
from enterpriseiq.agent.routing import AgentRouter
from enterpriseiq.agent.tools import ToolContext
from enterpriseiq.rag import GroundedRAGPipeline
from enterpriseiq.rag.models import GroundedResponse


class AgentOrchestrationError(RuntimeError):
    """Raised when the agent orchestration layer cannot execute safely."""


@runtime_checkable
class RAGAnswerer(Protocol):
    """Minimal grounded RAG interface required by the agent."""

    def answer(self, query: str) -> GroundedResponse:
        """Return a grounded response for the supplied query."""
        ...


@dataclass(frozen=True, slots=True)
class AgentOrchestratorConfig:
    """Configuration for deterministic agent orchestration."""

    tool_success_answer: str = (
        "Tool execution completed successfully."
    )
    tool_failure_answer: str = (
        "I could not complete the requested tool operation safely."
    )
    default_abstention_answer: str = (
        "I cannot complete that request with the available enterprise "
        "knowledge and approved tools."
    )

    def __post_init__(self) -> None:
        for field_name in (
            "tool_success_answer",
            "tool_failure_answer",
            "default_abstention_answer",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise AgentOrchestrationError(
                    f"{field_name} must be a non-empty string."
                )


class AgenticRAGOrchestrator:
    """Coordinate routing, grounded RAG, and controlled tool execution."""

    def __init__(
        self,
        *,
        router: AgentRouter,
        rag_pipeline: GroundedRAGPipeline,
        tool_executor: SafeToolExecutor,
        config: AgentOrchestratorConfig | None = None,
    ) -> None:
        if not isinstance(router, AgentRouter):
            raise AgentOrchestrationError(
                "router must satisfy the AgentRouter protocol."
            )

        if not isinstance(
            rag_pipeline,
            GroundedRAGPipeline,
        ):
            raise AgentOrchestrationError(
                "rag_pipeline must be a GroundedRAGPipeline."
            )

        if not isinstance(
            tool_executor,
            SafeToolExecutor,
        ):
            raise AgentOrchestrationError(
                "tool_executor must be a SafeToolExecutor."
            )

        if config is not None and not isinstance(
            config,
            AgentOrchestratorConfig,
        ):
            raise AgentOrchestrationError(
                "config must be an AgentOrchestratorConfig when provided."
            )

        self._router = router
        self._rag_pipeline = rag_pipeline
        self._tool_executor = tool_executor
        self._config = config or AgentOrchestratorConfig()

    @property
    def router(self) -> AgentRouter:
        """Return the configured routing layer."""
        return self._router

    @property
    def rag_pipeline(self) -> GroundedRAGPipeline:
        """Return the configured grounded RAG pipeline."""
        return self._rag_pipeline

    @property
    def tool_executor(self) -> SafeToolExecutor:
        """Return the controlled tool executor."""
        return self._tool_executor

    @property
    def config(self) -> AgentOrchestratorConfig:
        """Return the orchestration configuration."""
        return self._config

    def run(
        self,
        query: AgentQuery | str,
    ) -> AgentResponse:
        """Execute one complete agentic RAG request."""
        agent_query = self._normalize_query(query)

        try:
            decision = self._router.route(agent_query)
        except Exception as exc:
            raise AgentOrchestrationError(
                "Agent routing failed."
            ) from exc

        if decision.decision_type is DecisionType.DIRECT_RAG:
            return self._run_direct_rag(
                agent_query,
                decision,
            )

        if decision.decision_type is DecisionType.TOOL_CALL:
            return self._run_tool_call(
                agent_query,
                decision,
            )

        if decision.decision_type is DecisionType.ABSTAIN:
            return self._run_abstention(
                agent_query,
                decision,
            )

        raise AgentOrchestrationError(
            "Router returned an unsupported decision type."
        )

    def _run_direct_rag(
        self,
        query: AgentQuery,
        decision: AgentDecision,
    ) -> AgentResponse:
        """Execute the direct grounded-RAG path."""
        decision_step = AgentStep(
            step_number=1,
            action="route_direct_rag",
            decision=decision,
        )

        try:
            grounded = self._rag_pipeline.answer(
                query.text
            )
        except Exception as exc:
            return self._failed_response(
                query=query,
                decision=decision,
                steps=(decision_step,),
                answer=(
                    "I could not generate a grounded "
                    "enterprise answer."
                ),
                metadata={
                    "path": "direct_rag",
                    "failure_stage": "rag_pipeline",
                    "error_type": type(exc).__name__,
                },
            )

        rag_step = AgentStep(
            step_number=2,
            action="grounded_rag",
            metadata={
                "grounded": grounded.grounded,
                "abstained": grounded.abstained,
                "evidence_count": grounded.evidence_count,
                "citation_count": len(
                    grounded.citations
                ),
            },
        )

        trace = AgentTrace(
            steps=(
                decision_step,
                rag_step,
            ),
            metadata={
                "path": "direct_rag",
            },
        )

        if grounded.abstained:
            final_decision = self._derived_abstain_decision(
                reason=(
                    grounded.answer
                    or self._config.default_abstention_answer
                ),
                source_decision=decision,
                source="grounded_rag",
            )

            return AgentResponse(
                query=query,
                status=AgentStatus.ABSTAINED,
                answer=(
                    grounded.answer
                    or self._config.default_abstention_answer
                ),
                decision=final_decision,
                trace=trace,
                metadata=self._rag_metadata(
                    grounded
                ),
            )

        return AgentResponse(
            query=query,
            status=AgentStatus.COMPLETED,
            answer=grounded.answer,
            decision=decision,
            trace=trace,
            metadata=self._rag_metadata(
                grounded
            ),
        )

    def _run_tool_call(
        self,
        query: AgentQuery,
        decision: AgentDecision,
    ) -> AgentResponse:
        """Execute one explicitly routed enterprise tool."""
        if decision.tool_call is None:
            raise AgentOrchestrationError(
                "TOOL_CALL decision must contain a tool call."
            )

        decision_step = AgentStep(
            step_number=1,
            action="route_tool_call",
            decision=decision,
        )

        context = ToolContext(
            query_id=query.query_id,
            metadata=query.metadata,
        )

        try:
            result = self._tool_executor.execute(
                decision.tool_call,
                context=context,
            )
        except Exception as exc:
            return self._failed_response(
                query=query,
                decision=decision,
                steps=(decision_step,),
                answer=self._config.tool_failure_answer,
                metadata={
                    "path": "tool_call",
                    "failure_stage": "tool_executor",
                    "error_type": type(exc).__name__,
                },
            )

        tool_step = AgentStep(
            step_number=2,
            action="execute_tool",
            tool_result=result,
            metadata={
                "tool_name": result.tool_name,
                "tool_status": result.status.value,
            },
        )

        steps = (
            decision_step,
            tool_step,
        )

        if result.status is ToolExecutionStatus.SUCCESS:
            return self._successful_tool_response(
                query=query,
                decision=decision,
                result=result,
                steps=steps,
            )

        if result.status is ToolExecutionStatus.REJECTED:
            final_decision = self._derived_abstain_decision(
                reason=self._config.tool_failure_answer,
                source_decision=decision,
                source="tool_rejected",
            )

            return AgentResponse(
                query=query,
                status=AgentStatus.ABSTAINED,
                answer=self._config.tool_failure_answer,
                decision=final_decision,
                trace=AgentTrace(
                    steps=steps,
                    metadata={
                        "path": "tool_call",
                    },
                ),
                tool_results=(result,),
                metadata={
                    "path": "tool_call",
                    "tool_name": result.tool_name,
                    "tool_status": result.status.value,
                },
            )

        return AgentResponse(
            query=query,
            status=AgentStatus.FAILED,
            answer=self._config.tool_failure_answer,
            decision=decision,
            trace=AgentTrace(
                steps=steps,
                metadata={
                    "path": "tool_call",
                },
            ),
            tool_results=(result,),
            metadata={
                "path": "tool_call",
                "tool_name": result.tool_name,
                "tool_status": result.status.value,
            },
        )

    def _successful_tool_response(
        self,
        *,
        query: AgentQuery,
        decision: AgentDecision,
        result: ToolResult,
        steps: tuple[AgentStep, ...],
    ) -> AgentResponse:
        """Build a completed response for a successful tool call."""
        answer = self._tool_answer(result)

        return AgentResponse(
            query=query,
            status=AgentStatus.COMPLETED,
            answer=answer,
            decision=decision,
            trace=AgentTrace(
                steps=steps,
                metadata={
                    "path": "tool_call",
                },
            ),
            tool_results=(result,),
            metadata={
                "path": "tool_call",
                "tool_name": result.tool_name,
                "tool_status": result.status.value,
            },
        )

    def _run_abstention(
        self,
        query: AgentQuery,
        decision: AgentDecision,
    ) -> AgentResponse:
        """Return a safe response when the router abstains."""
        answer = (
            decision.reason.strip()
            if decision.reason
            else self._config.default_abstention_answer
        )

        step = AgentStep(
            step_number=1,
            action="abstain",
            decision=decision,
        )

        return AgentResponse(
            query=query,
            status=AgentStatus.ABSTAINED,
            answer=answer,
            decision=decision,
            trace=AgentTrace(
                steps=(step,),
                metadata={
                    "path": "abstain",
                },
            ),
            metadata={
                "path": "abstain",
            },
        )

    def _failed_response(
        self,
        *,
        query: AgentQuery,
        decision: AgentDecision,
        steps: tuple[AgentStep, ...],
        answer: str,
        metadata: Mapping[str, Any],
    ) -> AgentResponse:
        """Build a sanitized failed response."""
        return AgentResponse(
            query=query,
            status=AgentStatus.FAILED,
            answer=answer,
            decision=decision,
            trace=AgentTrace(
                steps=steps,
                metadata={
                    "path": metadata.get(
                        "path",
                        "unknown",
                    ),
                },
            ),
            metadata=metadata,
        )

    def _tool_answer(
        self,
        result: ToolResult,
    ) -> str:
        """Convert safe tool output into a deterministic answer."""
        if result.output is None:
            return self._config.tool_success_answer

        if isinstance(result.output, str):
            text = result.output.strip()

            return (
                text
                if text
                else self._config.tool_success_answer
            )

        return str(result.output)

    @staticmethod
    def _derived_abstain_decision(
        *,
        reason: str,
        source_decision: AgentDecision,
        source: str,
    ) -> AgentDecision:
        """Create a contract-valid final abstention decision."""
        return AgentDecision(
            decision_type=DecisionType.ABSTAIN,
            reason=reason,
            metadata={
                "source": source,
                "source_decision_type": (
                    source_decision.decision_type.value
                ),
            },
        )

    @staticmethod
    def _rag_metadata(
        response: GroundedResponse,
    ) -> dict[str, Any]:
        """Expose grounded-response audit metadata."""
        return {
            "path": "direct_rag",
            "grounded": response.grounded,
            "abstained": response.abstained,
            "evidence_count": response.evidence_count,
            "citation_count": len(
                response.citations
            ),
            "rag_citations": tuple(
                {
                    "source_id": citation.source_id,
                    "document_id": citation.document_id,
                    "title": citation.title,
                }
                for citation in response.citations
            ),
            "rag_metadata": dict(
                response.metadata
            ),
        }

    @staticmethod
    def _normalize_query(
        query: AgentQuery | str,
    ) -> AgentQuery:
        """Normalize supported query inputs."""
        if isinstance(query, AgentQuery):
            return query

        if isinstance(query, str):
            return AgentQuery(
                text=query
            )

        raise AgentOrchestrationError(
            "query must be an AgentQuery or string."
        )


def run_agent(
    *,
    query: AgentQuery | str,
    router: AgentRouter,
    rag_pipeline: GroundedRAGPipeline,
    tool_executor: SafeToolExecutor,
    config: AgentOrchestratorConfig | None = None,
) -> AgentResponse:
    """Convenience wrapper for one agentic RAG execution."""
    orchestrator = AgenticRAGOrchestrator(
        router=router,
        rag_pipeline=rag_pipeline,
        tool_executor=tool_executor,
        config=config,
    )

    return orchestrator.run(query)