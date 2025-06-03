import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Setup face detector and face encoder
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def load_face(filepath):
    img = Image.open(filepath).convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise ValueError("Face not detected in image.")
    return face.unsqueeze(0)  # Add batch dimension

def get_embedding(face_tensor):
    with torch.no_grad():
        embedding = resnet(face_tensor)
    return embedding

def compare_faces(embedding1, embedding2):
    sim = cosine_similarity(embedding1.numpy(), embedding2.numpy())[0][0]
    return sim

def show_images(img1_path, img2_path, similarity):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(Image.open(img1_path))
    axs[0].set_title("Person A")
    axs[1].imshow(Image.open(img2_path))
    axs[1].set_title("Person B\nSimilarity: {:.4f}".format(similarity))
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img1_path = "images/person_a.jpg"
    img2_path = "images/person_b.jpg"

    face1 = load_face(img1_path)
    face2 = load_face(img2_path)

    emb1 = get_embedding(face1)
    emb2 = get_embedding(face2)

    similarity = compare_faces(emb1, emb2)
    print(f"Cosine similarity: {similarity:.4f}")

    match = similarity > 0.6
    print("Match!" if match else "Not a match.")
    show_images(img1_path, img2_path, similarity)
