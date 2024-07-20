import torch
import clip
from PIL import Image

# 设置设备为CUDA（如果可用），否则为CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型和预处理器
model, preprocess = clip.load("ViT-B/32", device=device)

# 修改为图像的实际路径
image_path = "E:\E盘作业\教程\搭建\CLIP\CLIP-main\cat.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 使用自定义标签
custom_labels = ["a cat", "a cow", "a car", "a person", "a tree"]
text = clip.tokenize(custom_labels).to(device)

# 不计算梯度
with torch.no_grad():
    # 编码图像和文本
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # 计算图像和文本之间的相似度
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印概率分布
for label, prob in zip(custom_labels, probs[0]):
    print(f"Label: {label}, Probability: {prob}")
