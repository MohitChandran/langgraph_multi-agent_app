"""The email service package (plan 05).

CrewAI would otherwise send usage telemetry to its servers. Its switches are set here, the package's first lines, so
they apply before crewai is imported however this package is used (the service, a CLI, the tests). setdefault keeps a
value you set yourself.

Which switches: crewai_core/telemetry.py checks CREWAI_DISABLE_TELEMETRY / CREWAI_DISABLE_TRACKING (and
OTEL_SDK_DISABLED); CREWAI_TRACING_ENABLED=false turns off its own trace upload. NOT OTEL_SDK_DISABLED: that switches
off the whole OpenTelemetry SDK, which Langfuse (app/tracing.py) is built on. (Plan 05 named CREWAI_TELEMETRY_OPT_OUT,
which CrewAI 1.15 doesn't read; checked in the source while adding tracing.)
"""

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
