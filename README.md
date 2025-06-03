# 🛡️ Adversarial Face Recognition Attacks

This project demonstrates how to spoof a face recognition model using adversarial perturbations. Using `facenet-pytorch`, we simulate a real-world biometric attack scenario where one person's face is subtly altered to match the identity of another.

## 🌐 Overview

* **Model**: FaceNet (InceptionResnetV1, pretrained on VGGFace2)
* **Framework**: PyTorch
* **Attack Method**: Gradient-based cosine similarity optimization
* **Defenses**: JPEG Compression, Gaussian Blur

---

## ⚔️ Objective

To craft an adversarial example of **Person B** that is misclassified by the model as **Person A**, despite the two being visually distinct.

## 🧰 Project Structure

```
adversarial-face-spoofing/
├── face_recognition_model.py   # Embedding + similarity logic
├── adversarial_attack.py       # Spoof generation and evaluation
├── defense.py                  # Input transformations (JPEG, blur)
├── images/                     # Input face images
│   ├── person_a.jpg
│   ├── person_b.jpg
```

---

## 📊 Attack Results

| Step | Cosine Similarity | Comment                |
| ---- | ----------------- | ---------------------- |
| 0    | 0.18              | Low initial similarity |
| 100  | 1.00              | Perfect spoof          |

Final similarity scores:

```
Adversarial image:   1.0000
JPEG compressed:     0.4812
Gaussian blurred:    0.5377
```

✅ **Defenses were successful**: similarity dropped below 0.6 threshold.

---

## 🔒 How the Attack Works

1. Extract embeddings of Person A and Person B
2. Iteratively adjust Person B's pixels to match Person A's embedding
3. Use cosine similarity loss and optimize using Adam
4. Clamp pixel values to stay within valid image bounds

---

## 🛡️ Defenses Implemented

* **JPEG Compression**: Removes minor pixel-level noise by re-encoding the image
* **Gaussian Blur**: Smooths input to degrade high-frequency adversarial perturbations

---

## ⚡ Setup

```bash
pip install facenet-pytorch foolbox torch torchvision numpy matplotlib opencv-python scikit-learn
```

---

## 🔧 Run the Attack

```bash
python adversarial_attack.py
```

---

## 🌐 Author

Built by Gregory Apostle as part of an AI Security portfolio project.