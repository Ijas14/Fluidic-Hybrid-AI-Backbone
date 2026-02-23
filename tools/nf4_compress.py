import torch
from transformers import AutoModelForCausalLM
from quark.torch import ModelQuantizer

# The Modern Quark API Imports
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization.config.config import QConfig, QTensorConfig, QLayerConfig
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

model_path = "DeepSeek-R1-1.5B-HF"

print("Step 1: Loading raw weights into CPU RAM...")
# Load to CPU first to protect our 4GB VRAM
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Step 2: Configure the 4-bit "Fluidic" Compression
print("Step 2: Configuring the QTensor Spec...")
spec = QTensorConfig(
    dtype=Dtype.int4, 
    observer_cls=PerGroupMinMaxObserver,
    symmetric=False,             # NF4 distribution is asymmetric
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_group, # This gives you the high-precision "fluidic" feel
    ch_axis=0,
    is_dynamic=False,
    group_size=128 
)

# Wrap it in the modern QConfig
quant_config = QConfig(
    global_quant_config=QLayerConfig(weight=spec)
)

print("Step 3: Applying 4-bit Quantization (ROCm 7.2 Stack)...")
quantizer = ModelQuantizer(quant_config)
# Crush the weights
quantized_model = quantizer.quantize_model(model, dataloader=None) 

print("Step 4: Saving the Fluidic Backbone...")
quantized_model.save_pretrained("models/DeepSeek-R1-1.5B-NF4")
print("BACKBONE READY: The logic engine is officially compressed.")

from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

print("Step 5: Exporting to GGUF for N.O.V.A...")
# Initialize the exporter
config = ExporterConfig(json_export_config=JsonExporterConfig())
exporter = ModelExporter(config=config, export_dir="models/")

# Export the quantized model
from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

print("Step 5: Exporting to GGUF for N.O.V.A...")
# Initialize the exporter
config = ExporterConfig(json_export_config=JsonExporterConfig())
exporter = ModelExporter(config=config, export_dir="models/")

from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

print("Step 5: Exporting to GGUF for N.O.V.A...")
# Initialize the exporter
config = ExporterConfig(json_export_config=JsonExporterConfig())
exporter = ModelExporter(config=config, export_dir="models/")

# Export the quantized model
exporter.export_gguf_model(
    model=quantized_model, 
    model_dir=model_path,     # Path to the original HF folder for metadata
    model_type="llama"
)

print("PACKAGING COMPLETE: NOVA-Fluidic-1.5B.gguf is ready.")
