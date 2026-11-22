"""The agent card: this service's public "business card" in A2A (plan 05, section 5).

Any A2A client reads it from GET /.well-known/agent-card.json BEFORE talking to the agent, and learns from it:
where to send messages (the interface URL + protocol binding), which protocol version, what the agent can do (its
skills), what data it takes and returns, and which authentication it expects. The orchestrator's client builds itself
from this card (ClientFactory.create_from_url), so the card must be accurate.
"""

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    APIKeySecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)

from app.config import Settings

API_KEY_HEADER = "X-API-Key"
API_KEY_SCHEME = "apiKey"


def build_agent_card(settings: Settings) -> AgentCard:
    """The card for this deployment (its public URL comes from EMAIL_SERVICE_PUBLIC_URL)."""
    return AgentCard(
        name="Email & Dispatch Agent",
        description=(
            "Drafts a business email with a CrewAI crew (writer, then compliance reviewer), pauses for human approval "
            "(task state INPUT_REQUIRED), and sends it over SMTP only after an explicit approval."
        ),
        version="0.1.0",
        provider=AgentProvider(organization="Ops Platform", url=settings.email_service_public_url),
        supported_interfaces=[
            AgentInterface(
                url=settings.email_service_public_url.rstrip("/") + "/",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="draft_and_send_email",
                name="Draft and send an email",
                description=(
                    "Drafts an email from the given content, waits for human approval, then sends it to exactly one "
                    'recipient. Message 1: data {"to", "subject", "content", "request"}. Message 2 on the same task: '
                    'data {"decision": "approve" | "reject", "edited_body"}.'
                ),
                tags=["email", "approval"],
                examples=['{"to": "accounts@company.example", "subject": "Pending invoices", "content": "..."}'],
            )
        ],
        # Declares the API key: clients learn from the card that every call needs the X-API-Key header.
        security_schemes={
            API_KEY_SCHEME: SecurityScheme(
                api_key_security_scheme=APIKeySecurityScheme(
                    location="header", name=API_KEY_HEADER, description="Shared key between orchestrator and service"
                )
            )
        },
        security_requirements=[SecurityRequirement(schemes={API_KEY_SCHEME: StringList()})],
    )
