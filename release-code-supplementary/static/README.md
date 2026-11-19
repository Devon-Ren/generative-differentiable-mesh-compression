# ByteMesh

Code for "ByteMesh: Rate-Controlled Training and Compression of Opaque Meshes".

## Contents

| Path | Purpose |
|---|---|
| `training/train_bytemesh.py` | Rate-controlled training |
| `training/rd_growth.py` | Per-triangle error and split selection |
| `training/rd_prune.py` | Edge collapse |
| `training/size_probe.py` | Size probe |
| `postprocess/` | Conversion of the trained model and fine-tuning before coding |
| `encode.py` | Encoder: codebooks, refinement, compressed file |
| `decode.py` | Decoder: compressed file to indexed triangle mesh |
| `codec/codec/` | BM-Codec: quantization, vector quantization, lossless coding |
| `codec/` | Mesh model with shared codebooks and its renderer |
| `run_pipeline.sh` | Training, encoding and decoding of one scene |

## Requirements

The public MeshSplatting code, installed as described in its README and placed at `<LAB>/upstream`.

```bash
pip install -r requirements.txt
bash build.sh
```

## Usage

```bash
bash run_pipeline.sh <LAB> <DATASET> <IMAGES> <TARGET_MB> <POSITION_BITS> <OUTPUT> [INDOOR]
```

```bash
bash run_pipeline.sh /path/to/lab /path/to/mipnerf360/garden images_4 10 q15 /path/to/output 0
```

```bash
python decode.py --asset <FILE> --out mesh.pt
```

## License

`codec/renderer/` is derived from third-party code; its licenses are in that directory.
