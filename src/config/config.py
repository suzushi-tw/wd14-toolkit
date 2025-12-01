# PixAI Tagger
PIXAI_TAGGER_V09_REPO = "pixai-labs/pixai-tagger-v0.9"

# Dataset v3 series of models
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"
VIT_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-large-tagger-v3"
EVA02_LARGE_MODEL_DSV3_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"

# Dataset v2 series of models
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# IdolSankaku series of models
SWINV2_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-swinv2-tagger-v1"
EVA02_LARGE_MODEL_IS_DSV1_REPO = "deepghs/idolsankaku-eva02-large-tagger-v1"

dropdown_list = [
    SWINV2_MODEL_DSV3_REPO,  # Recommended: Best balance of speed and accuracy
    PIXAI_TAGGER_V09_REPO,  # PyTorch model with 13k tags
    CONV_MODEL_DSV3_REPO,
    VIT_MODEL_DSV3_REPO,
    VIT_LARGE_MODEL_DSV3_REPO,
    EVA02_LARGE_MODEL_DSV3_REPO,
    MOAT_MODEL_DSV2_REPO,
    SWIN_MODEL_DSV2_REPO,
    CONV_MODEL_DSV2_REPO,
    CONV2_MODEL_DSV2_REPO,
    VIT_MODEL_DSV2_REPO,
    SWINV2_MODEL_IS_DSV1_REPO,
    EVA02_LARGE_MODEL_IS_DSV1_REPO,
]