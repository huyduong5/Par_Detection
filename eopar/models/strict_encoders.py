"""Offline-only, fail-closed pretrained loaders for the isolated EOPAR model."""
import json
from dataclasses import fields
from pathlib import Path

import torch
from aq_visibility.model.vision_encoder import VMambaVisionEncoder as BaseVision
from aq_visibility.model.text_encoder import MambaTextEncoder as BaseText
from eopar.train.runtime import ROOT, digest
from eopar.models.weights import checked_state


class VMambaVisionEncoder(BaseVision):
    def _load_pretrained(self, pretrained_path=None):
        path = Path(pretrained_path) if pretrained_path else ROOT / 'aq_visibility/checkpoints' / f'vmamba_{self.variant}_pretrained.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Required offline vision checkpoint: {path}')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        checked_state(self.backbone, payload.get('model', payload))
        self.pretrained_files = {str(path.resolve()): digest(path)}


class MambaTextEncoder(BaseText):
    def _load_model_and_tokenizer(self):
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from safetensors.torch import load_file
        config_path = Path(hf_hub_download(self.hf_name, 'config.json', local_files_only=True))
        weights_path = Path(hf_hub_download(self.hf_name, 'model.safetensors', local_files_only=True))
        tokenizer_path = Path(snapshot_download('EleutherAI/gpt-neox-20b', local_files_only=True))
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg = json.loads(config_path.read_text())
        keys = {f.name for f in fields(MambaConfig)}
        model = MambaLMHeadModel(MambaConfig(**{k: v for k, v in cfg.items() if k in keys}),
                                device='cpu', dtype=torch.float32)
        state = load_file(str(weights_path), device='cpu')
        if 'backbone.embeddings.weight' in state:
            state['backbone.embedding.weight'] = state.pop('backbone.embeddings.weight')
        allowed = ()
        if model.lm_head.weight.data_ptr() == model.backbone.embedding.weight.data_ptr():
            allowed = ('lm_head.weight',)
        checked_state(model, state, allowed)
        files = [config_path, weights_path] + [p for p in tokenizer_path.iterdir()
                 if p.suffix in ('.json', '.txt') and p.is_file()]
        self.pretrained_files = {str(p.resolve()): digest(p) for p in files}
        return model, tokenizer
