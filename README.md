## Model Architecture

The segmentation model used in this project is a **U-Net with a MobileNetV2 CNN backbone** designed for **medical image segmentation** of the Nuchal Translucency (NT) region in ultrasound images.

### Encoder (CNN Feature Extractor)

The encoder of the network uses **MobileNetV2**, a lightweight **Convolutional Neural Network (CNN)** that extracts hierarchical image features.

```python
base_model = tf.keras.applications.MobileNetV2(
    input_tensor=x,
    include_top=False,
    weights=None
)
```

This encoder learns important ultrasound features at multiple spatial scales.
Intermediate feature maps from the network are used as **skip connections**, which help the decoder recover spatial details lost during downsampling.

```python
s1 = base_model.get_layer("block_1_expand_relu").output
s2 = base_model.get_layer("block_3_expand_relu").output
s3 = base_model.get_layer("block_6_expand_relu").output
s4 = base_model.get_layer("block_13_expand_relu").output
```

These multi-scale features are a key component of the **U-Net architecture**.

---

### Bridge (Bottleneck Layer)

The deepest representation of the image acts as the **bridge between encoder and decoder**.

```python
bridge = base_model.get_layer("block_16_project").output
```

This layer captures high-level semantic features before the upsampling process begins.

---

### Decoder (U-Net Upsampling Path)

The decoder reconstructs the segmentation mask by progressively upsampling the feature maps and combining them with encoder features through skip connections.

```python
Conv2DTranspose(...)
concatenate([d1, s4])
```

Each decoder stage follows the pattern:

```
Upsample
↓
Concatenate with skip connection
↓
Convolution layers
```

This allows the network to restore fine spatial details while preserving contextual information from the encoder.

---

### Final Segmentation Output

The final layer generates the segmentation mask:

```python
outputs = Conv2D(1, 1, activation="sigmoid")(d5)
```

The output is a:

```
256 × 256 × 1 segmentation mask
```

Each pixel represents a **probability (0–1)** of belonging to the **Nuchal Translucency region**.

---

### Model Architecture Overview

```
Input (256 × 256 × 1)
        │
1×1 Conv → Convert grayscale to RGB
        │
MobileNetV2 Encoder (CNN)
        │
   ┌────┴────┐
   │  Bridge │
   └────┬────┘
        │
   U-Net Decoder
        │
Upsampling + Skip Connections
        │
Convolution Layers
        │
Output Segmentation Mask
```

---

### Why This Architecture Works Well

This design is particularly effective for **medical ultrasound segmentation**:

| Component        | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| CNN Encoder      | Learns ultrasound image features                  |
| U-Net Decoder    | Enables precise segmentation boundaries           |
| Skip Connections | Preserve spatial details lost during downsampling |

The combination of a **CNN encoder with a U-Net decoder** is widely used in medical imaging research due to its strong performance on small datasets and fine boundary detection.

---

### Model File

The trained model file **`best_model.keras`** exceeds GitHub’s file size limit for repository uploads.

You can download the trained model from the **Releases section of this repository**.
