from contextlib import contextmanager

from evalrag.models.langfuse import LangfuseObservation
# [STUB] En production : client Langfuse partagé (rag.services.observability_service)
from evalrag.stubs.langfuse_client import get_langfuse_client


class EvaluationTracer:
    def __init__(self):
        self.langfuse = get_langfuse_client()

    @contextmanager
    def observe(self, obs: LangfuseObservation):
        with self.langfuse.start_as_current_observation(**obs.to_langfuse()) as span:
            if obs.as_type == "chain" and obs.input and obs.input.chat_id:
                span.update_trace(session_id=obs.input.chat_id)
            yield span

        # -------------------------
        # Scores (par interaction)
        # -------------------------
        def log_scores(self, scores: dict[str, float | None]):
            if not self.root_obs:
                return

            for metric, value in scores.items():
                if value is None:
                    continue

                self.langfuse.create_score(
                    name=metric,
                    value=value,
                    trace_id=self.root_obs.trace_id,
                    observation_id=self.root_obs.id,
                    data_type="NUMERIC",
                )
