"""Concrete model registry entries.

Whisper weights are managed by faster-whisper's HuggingFace cache (pointed
at PUBLIKCLIP_HOME/models/hf via HF_HOME in the ASR stage); everything else
is fetched explicitly through the registry so the app can show one honest
download progress list. All entries are ungated — no tokens, no accounts
(the CAM++ Apache-2.0 verification is what made that possible).

Every URL below addresses an *immutable* revision — a commit sha, a HF
revision sha, or a Zenodo record — never a branch name. `raw/main` and
`resolve/main` resolve to whatever the upstream owner pushed last, which
hands a third-party account silent write access to weights this app
deserializes; a pickle load is arbitrary code execution. The pinned sha256
is the real enforcement (registry.ensure verifies it), the revision pin just
means a rewritten branch fails loudly at download instead of at checksum.

To refresh a pin: fetch from the new revision, hash it, and update BOTH the
URL and the sha256 in the same edit.
"""

from .registry import ModelSpec, register

# Upstream revisions these weights were pinned at (2026-08-21).
_CLIP_FORGE = "d7653b460ea707232180a86fd266f90aef6022d1"
_LAUGHTER_DETECTION = "5d5e0327916959d832d95ffbef5f484efc93d799"
_CAMPPLUS_HF = "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"

LAUGHTER = register(
    ModelSpec(
        name="laughter-jrgillick",
        filename="best.pth.tar",
        url=(
            f"https://github.com/jrgillick/laughter-detection/raw/{_LAUGHTER_DETECTION}/"
            "checkpoints/in_use/resnet_with_augmentation/best.pth.tar"
        ),
        sha256="bfe450e41926a4e9de2abf007c9a13fa8420439eaa1383e986563c565f5ef206",
        approx_mb=10,
    )
)

# Zenodo records are immutable by design (the DOI covers the bytes), so the
# record id is itself the revision pin.
PANNS_CNN14_MAX = register(
    ModelSpec(
        name="panns-cnn14-decisionlevelmax",
        filename="Cnn14_DecisionLevelMax.pth",
        url=(
            "https://zenodo.org/record/3987831/files/"
            "Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
        ),
        sha256="dd3b4043a87d4ec13df8082c0fcfee3fb5084151808e47e060987a95eabdd142",
        approx_mb=312,
    )
)

CAMPPLUS = register(
    ModelSpec(
        name="campplus",
        filename="campplus_cn_common.bin",
        url=(
            f"https://huggingface.co/funasr/campplus/resolve/{_CAMPPLUS_HF}/"
            "campplus_cn_common.bin"
        ),
        sha256="3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
        approx_mb=28,
    )
)

# clip-forge ships these pre-exported (MIT); its export-asd-onnx.py proves
# numerical parity against the LR-ASD reference implementation.
ULTRAFACE = register(
    ModelSpec(
        name="ultraface",
        filename="ultraface-rfb-320.onnx",
        url=(
            f"https://github.com/JeremySNR/clip-forge/raw/{_CLIP_FORGE}/"
            "resources/models/ultraface-rfb-320.onnx"
        ),
        sha256="34cd7e60aeff28744c657de7a3dc64e872d506741de66987f3426f2b79f88017",
        approx_mb=2,
    )
)

LR_ASD_FRONTEND = register(
    ModelSpec(
        name="lr-asd",
        filename="frontend.onnx",
        url=(
            f"https://github.com/JeremySNR/clip-forge/raw/{_CLIP_FORGE}/"
            "resources/models/lr-asd-frontend.onnx"
        ),
        sha256="f7c055612cd6f1f2da3ab8257567ab68a6b0d69b5e436699a5cf65334dd79461",
        approx_mb=3,
    )
)

LR_ASD_BACKEND = register(
    ModelSpec(
        name="lr-asd",
        filename="backend.onnx",
        url=(
            f"https://github.com/JeremySNR/clip-forge/raw/{_CLIP_FORGE}/"
            "resources/models/lr-asd-backend.onnx"
        ),
        sha256="9453caa09998027995664fd5a3b1fab4ad0de30a92c6beba8c29c3619de510a9",
        approx_mb=1,
    )
)
