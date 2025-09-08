# TFLite Segmentation (Android)

**O que é?**  
App Android que faz **segmentação semântica** (gato/cão) em imagens escolhidas pelo usuário (câmera/galeria), usando **TensorFlow Lite** com um **UNet FP32**. A saída é sobreposta à imagem original como uma máscara vermelha semitransparente.

---

## Screenshots
<p align="center">
  <img src="https://github.com/user-attachments/assets/034ce543-baa9-44aa-8cee-ae4ae6b7d0d0" alt="cachorros" width="24%" />
  <img src="https://github.com/user-attachments/assets/4b560176-ad55-4703-90df-6bda768d20ee" alt="gatos" width="24%" />
</p>

---

## 🔧 Requisitos

- Android Studio com **JDK 17**
- API mínima 24 (Android 7.0)
- Dispositivo físico recomendado para melhor desempenho

---

## O que vem no app

- **Entrada:** imagem redimensionada para `256×256`, normalizada (`0..1`)
- **Inferência:** TensorFlow Lite **FP32**
- **Pós-processamento:** `sigmoid(logit)` + **threshold = 0.5**  
- **Exibição:** overlay **vermelho** sobre a imagem original

---

## Estrutura essencial

- `MainActivity.kt` – fluxo de UI (seleção da imagem, botão **Predict**)  
- `SegmentationInterpreter.kt` – wrapper do **Interpreter** TFLite  
- `activity_main.xml` – layout com `ImageView` da imagem e do overlay  
- `assets/unet_pet_simp_float32.tflite` – modelo a ser carregado
