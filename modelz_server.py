"""
envd build -f :serving --output type=image,name=ghcr.io/cosmicbboy/modelz-stable-diffusion:v1,push=true
"""

from io import BytesIO
from typing import List

import torch  # type: ignore
from diffusers import StableDiffusionPipeline  # type: ignore
from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin

logger = get_logger()


class StableDiffusion(MsgpackMixin, Worker):
    def __init__(self):
        self.pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(device)

    def forward(self, data: List[str]) -> List[memoryview]:
        logger.debug("generate images for %s", data)
        res = self.pipe(data)
        logger.debug("NSFW: %s", res[1])
        images = []
        for img in res[0]:
            dummy_file = BytesIO()
            img.save(dummy_file, format="JPEG")
            images.append(dummy_file.getbuffer())
        return images


if __name__ == "__main__":
    server = Server()
    server.append_worker(StableDiffusion, num=1, max_batch_size=4)
    server.run()
