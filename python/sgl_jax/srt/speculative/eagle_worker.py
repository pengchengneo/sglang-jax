from typing import Optional, Tuple

import jax

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.spec_info import (
    EagleDraftInput,
    EagleVerifyInput,
    SpeculativeAlgorithm,
)


class EAGLEWorker(ModelWorker):
    def __init__(self, server_args, targer_worker: ModelWorker):
        self.target_worker = targer_worker
        pass

    def forward_batch_speculative_generation(
        self,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
        sample_meta_data: SamplingMetadata,
    ):
        # prefill : Target Extend -> Decode Extend for Update Draft State
        # Decode : Draft → Verify → Update Draft State → Draft → Verify → ...

        if batch.forward_mode.is_extend():
            # target extend
            logits_output, next_token_ids, cache_miss_count, bid, seq_lens = (
                self.forward_target_extend(model_worker_batch, sample_meta_data)
            )
            # draft extend for Update Draft State
            self.forward_draft_extend(
                batch, logits_output.hidden_states, next_token_ids, seq_lens
            )
            return logits_output, next_token_ids, cache_miss_count
        else:
            # draft
            spec_info = self.draft(batch)
            # verify
            logits_output, verify_output, model_worker_batch, cache_hit = self.verify(
                batch, spec_info
            )
            self.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                next_token_ids,
                cache_miss_count,
                sum(verify_output.accept_length_per_req_cpu),
            )

    def forward_target_extend(
        self, model_worker_batch: ModelWorkerBatch, sample_meta_data: SamplingMetadata
    ) -> Tuple[LogitsProcessorOutput, jax.Array, int, int, jax.Array]:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sample_meta_data
            )
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
        seq_lens_cpu: Optional[jax.Array],
    ):
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            next_token_ids=next_token_ids,
            seq_lens_cpu=seq_lens_cpu,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        logits_output, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(
                model_worker_batch, self.mesh
            ),
        )
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)
        has_finished, unfinished_req_index = False, []
        for i, req in enumerate(batch.reqs):
            if req.finished():
                has_finished = True
            else:
                unfinished_req_index.append(i)
        if has_finished:
            unfinished_index_device = jax.numpy.array(
                unfinished_req_index,
                dtype=jax.numpy.int64,
                device=batch.spec_info.topk_p.device,
            )
            batch.spec_info.filter_batch(
                unfinished_index_device, has_been_filtered=False
            )

    @property
    def draft_model_runner(self):
        return self.target_worker.get_model_runner()

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, spec_info: EagleDraftInput
    ):
        pass

    def draft(self, batch: ScheduleBatch):
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleDraftInput):
        pass

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        pass
