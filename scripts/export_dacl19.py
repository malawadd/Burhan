"""Safely extract author DACL SegFormer weights and export a local ONNX graph.

Use only in isolated export environment. --extract-only performs no inference.
The default export performs Torch forward passes and requires a compute window.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'models/construction/dacl19'
EXPECTED = '166b778bdbc57ef2bfd5f71bc6ae724bf598d1a4c6a1b69afa4fbd936f2d965b'
REVIEWED_GLOBALS = {
    'torch._C._nn.gelu', 'transformers.activations.GELUActivation',
    'transformers.models.segformer.configuration_segformer.SegformerConfig', '__main__.SegModel',
    *['transformers.models.segformer.modeling_segformer.' + name for name in [
        'SegformerDWConv', 'SegformerOverlapPatchEmbeddings', 'SegformerModel', 'SegformerAttention',
        'SegformerDecodeHead', 'SegformerMixFFN', 'SegformerDropPath', 'SegformerEncoder',
        'SegformerSelfOutput', 'SegformerLayer', 'SegformerEfficientSelfAttention',
        'SegformerForSemanticSegmentation', 'SegformerMLP']],
    *['torch.nn.modules.' + name for name in ['conv.Conv2d', 'batchnorm.BatchNorm2d', 'dropout.Dropout',
        'container.ModuleList', 'normalization.LayerNorm', 'linear.Linear', 'linear.Identity', 'activation.ReLU', 'upsampling.Upsample']],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--size', type=int, choices=[320, 512], default=512)
    args = parser.parse_args()
    import torch
    from torch import nn
    torch.set_num_threads(2)
    import transformers
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    class SegModel(nn.Module):
        def forward(self, x):
            return self.upsample(self.segformer(x).logits)
    original = DIRECTORY / 'author-segformer-mit-b1.pth'
    if hashlib.sha256(original.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError('Author model hash mismatch')
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(original))
    if not unsafe.issubset(REVIEWED_GLOBALS):
        raise RuntimeError(f'Unreviewed checkpoint globals: {unsafe - REVIEWED_GLOBALS}')
    allowed = []
    for symbol in sorted(unsafe):
        if symbol == '__main__.SegModel':
            allowed.append((SegModel, symbol))
        else:
            module, name = symbol.rsplit('.', 1)
            allowed.append(getattr(importlib.import_module(module), name))
    # Explicit weights-only unpickler; never execute arbitrary checkpoint globals.
    with torch.serialization.safe_globals(allowed):
        restored = torch.load(original, map_location='cpu', weights_only=True)
    if not isinstance(restored, SegModel):
        raise RuntimeError('Unexpected original wrapper type')
    config = restored.segformer.config.to_dict()
    ordered_labels = ['Crack', 'ACrack', 'Wetspot', 'Efflorescence', 'Rust', 'Rockpocket', 'Hollowareas', 'Cavity', 'Spalling', 'Graffiti', 'Weathering', 'Restformwork', 'ExposedRebars', 'Bearing', 'EJoint', 'Drainage', 'PEquipment', 'JTape', 'WConccor']
    if [config['id2label'].get(index, config['id2label'].get(str(index))) for index in range(19)] != ordered_labels:
        raise RuntimeError('Author checkpoint label order does not match the dataset')
    # The 2023 full-model pickle stores a Torch dtype object. Export rebuilds
    # in explicit float32, so avoid carrying legacy version-specific dtype state.
    config.pop('torch_dtype', None)
    config.pop('dtype', None)
    state = {key: value.detach().clone() for key, value in restored.segformer.state_dict().items()}
    state_path = DIRECTORY / 'segformer-state-dict.pt'
    torch.save(state, state_path)
    (DIRECTORY / 'segformer-config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    report = {'source_sha256': EXPECTED, 'extraction': 'Torch weights_only=True plus statically inspected reviewed type allowlist; no weights_only=False',
              'reviewed_globals': sorted(unsafe), 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'state_sha256': hashlib.sha256(state_path.read_bytes()).hexdigest(), 'tensor_count': len(state),
              'parameters': sum(value.numel() for value in state.values()), 'ordered_author_labels': ordered_labels,
              'model_forward_performed': not args.extract_only}
    if not args.extract_only:
        network = SegformerForSemanticSegmentation(SegformerConfig(**config)).eval()
        network.load_state_dict(state, strict=True)
        class Export(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
            def forward(self, bgr):
                rgb = bgr[:, [2, 1, 0]].to(torch.float32) / 255
                return self.model((rgb - self.mean) / self.std).logits
        wrapped = Export(network).eval()
        target = DIRECTORY / f'dacl-segformer-b1-{args.size}.onnx'
        # Output retains native stride-4 logits. Sigmoid per class, never argmax.
        with torch.no_grad():
            torch.onnx.export(wrapped, (torch.zeros(1, 3, args.size, args.size, dtype=torch.uint8),), str(target),
                              input_names=['bgr'], output_names=['logits'], opset_version=17,
                              dynamo=False, do_constant_folding=True)
        report.update(onnx=str(target), onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                      input_shape=[1, 3, args.size, args.size], output_shape=[1, 19, args.size // 4, args.size // 4],
                      preprocessing='Externally stretch-resize to fixed size; uint8 NCHW BGR. RGB/ImageNet normalization inside graph.',
                      output_semantics='19 independent multilabel logits; sigmoid threshold per channel; nearest upsample preserves author demo output semantics. Not softmax/argmax.')
    (DIRECTORY / ('safe-extraction.json' if args.extract_only else f'export-{args.size}.json')).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
