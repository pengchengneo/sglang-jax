from sgl_jax.srt.managers.tp_worker import ModelWorker


class EAGLEWorker(ModelWorker):
    def __init__(
        self, server_args, mesh, is_draft_worker=False, req_to_token_pool=None
    ):
        super().__init__(server_args, mesh, is_draft_worker, req_to_token_pool)
