# D:\diffusers\models\diffuser\real

from diffusers import StableDiffusionPipeline
import torch
if __name__ == "__main__":
    model_id = "D:/diffusers/models/diffuser/realz"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
    pipe = pipe.to("cuda")

    prompt = "1girl"
    image = pipe(prompt).images[0]
    image
    # 保存为彩色图片
    image.save("image.jpg")
