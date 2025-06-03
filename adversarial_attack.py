import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from facenet_pytorch import InceptionResnetV1
from face_recognition_model import load_face, get_embedding, compare_faces
from defense import apply_jpeg_compression, apply_gaussian_blur

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Load face images
face_b = load_face("images/person_b.jpg").to(device)
face_a = load_face("images/person_a.jpg").to(device)

# Get target embedding (Person A)
target_embedding = get_embedding(face_a).detach()

# === Step 1: Adversarial Spoofing ===
face_b_adv = face_b.clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([face_b_adv], lr=0.01)

print("🔁 Starting adversarial spoofing...")
for step in range(101):
    optimizer.zero_grad()
    emb_spoof = resnet(face_b_adv)
    loss = 1 - torch.nn.functional.cosine_similarity(emb_spoof, target_embedding).mean()
    loss.backward()
    optimizer.step()
    face_b_adv.data = torch.clamp(face_b_adv.data, 0, 1)

    if step % 10 == 0 or step == 100:
        sim = torch.nn.functional.cosine_similarity(emb_spoof, target_embedding).item()
        print(f"Step {step:03d}: Cosine similarity = {sim:.4f}, Loss = {loss.item():.4f}")

# === Step 2: Evaluate Defenses ===
jpeg_defended = apply_jpeg_compression(face_b_adv, quality=50)
blurred_defended = apply_gaussian_blur(face_b_adv, ksize=5)

sim_orig_adv = compare_faces(get_embedding(face_a), get_embedding(face_b_adv))
sim_jpeg = compare_faces(get_embedding(face_a), get_embedding(jpeg_defended))
sim_blur = compare_faces(get_embedding(face_a), get_embedding(blurred_defended))

print("\n🎯 Final Similarities:")
print(f"  Adversarial image:   {sim_orig_adv:.4f}")
print(f"  JPEG compressed:     {sim_jpeg:.4f}")
print(f"  Gaussian blurred:    {sim_blur:.4f}")

# === Step 3: Visualize Images ===
def tensor_to_img(tensor):
    return tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

orig_img = tensor_to_img(face_b)
adv_img = tensor_to_img(face_b_adv)
jpeg_img = tensor_to_img(jpeg_defended)
blur_img = tensor_to_img(blurred_defended)

# Plot
plt.figure(figsize=(10, 4))
titles = [
    "Original Person B",
    "Spoofed (Person A)",
    f"JPEG Defense\nSim: {sim_jpeg:.2f}",
    f"Blur Defense\nSim: {sim_blur:.2f}"
]
images = [orig_img, adv_img, jpeg_img, blur_img]

for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(np.clip(images[i], 0, 1))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
