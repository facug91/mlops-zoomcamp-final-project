from PIL import Image
import matplotlib.pyplot as plt

fruit_types = ["apple", "banana", "grape", "strawberry", "mango"]
images_paths = [ f"./fruit_images/{fruit}.jpg" for fruit in fruit_types ]

image_size = (256, 256)  # Tamaño uniforme para todas las imágenes

fig, axes = plt.subplots(1, len(images_paths), figsize=(len(images_paths) * 2, 2))
axes = axes.flatten()

for ax, img_path, fruit_type in zip(axes, images_paths, fruit_types):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(image_size)
    
    ax.imshow(img)
    ax.set_title(fruit_type, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("fruit_banner.png", dpi=150)
plt.close(fig)
