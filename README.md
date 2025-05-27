# OmniConsistency

> **OmniConsistency: Learning Style-Agnostic
Consistency from Paired Stylization Data**
> <br>
> [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), 
> [Cheng Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=TvdVuAYAAAAJ), 
> and 
> [Mike Zheng Shou](https://sites.google.com/view/showlab)
> <br>
> [Show Lab](https://sites.google.com/view/showlab), National University of Singapore
> <br>

<a href="https://arxiv.org/abs/2505.18445"><img src="https://img.shields.io/badge/ariXv-2505.18445-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/spaces/yiren98/OmniConsistency"><img src="https://img.shields.io/badge/🤗_HuggingFace-Space-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/showlab/OmniConsistency"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>


<img src='./figure/teaser.png' width='100%' />

## Installation

We recommend using Python 3.10 and PyTorch with CUDA support. To set up the environment:

```bash
# Create a new conda environment
conda create -n omniconsistency python=3.10
conda activate omniconsistency

# Install other dependencies
pip install -r requirements.txt
```

## Download

You can download the OmniConsistency model and pretrained LoRAs directly from [Hugging Face](https://huggingface.co/showlab/OmniConsistency).
Or download using Python script:

### OmniConsistency Model

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/3D_Chibi_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/American_Cartoon_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Chinese_Ink_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Clay_Toy_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Fabric_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Ghibli_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Irasutoya_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Jojo_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/LEGO_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Line_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Macaron_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Oil_Painting_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Origami_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Paper_Cutting_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Picasso_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Pixel_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Poly_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Pop_Art_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Rick_Morty_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Snoopy_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Van_Gogh_rank128_bf16.safetensors", local_dir="./LoRAs")
hf_hub_download(repo_id="showlab/OmniConsistency", filename="LoRAs/Vector_rank128_bf16.safetensors", local_dir="./LoRAs")
```
### Pretrained LoRAs
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="showlab/OmniConsistency", filename="OmniConsistency.safetensors", local_dir="./Model")
```

## Usage
Here's a basic example of using OmniConsistency:

### Model Initialization
```python
import time
import torch
from PIL import Image
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Initialize model
device = "cuda"
base_path = "/path/to/black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16).to("cuda")

# Load OmniConsistency model
set_single_lora(pipe.transformer, 
                "/path/to/OmniConsistency.safetensors", 
                lora_weights=[1], cond_size=512)

# Load external LoRA
pipe.unload_lora_weights()
pipe.load_lora_weights("/path/to/lora_folder", 
                       weight_name="lora_name.safetensors")
```

### Style Inference
```python
image_path1 = "figure/test.png"
prompt = "3D Chibi style, Three individuals standing together in the office."

subject_images = []
spatial_image = [Image.open(image_path1).convert("RGB")]

width, height = 1024, 1024

start_time = time.time()

image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(5),
    spatial_images=spatial_image,
    subject_images=subject_images,
    cond_size=512,
).images[0]

end_time = time.time()
elapsed_time = end_time - start_time
print(f"code running time: {elapsed_time} s")

# Clear cache after generation
clear_cache(pipe.transformer)

image.save("results/output.png")
```

<!-- ## Citation
```

``` -->
