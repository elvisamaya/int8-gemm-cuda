# GPU vs CPU vs my accelerator

| Platform | Implementation | Throughput (GOPS) | Notes |
|---|---|---:|---|
| Tesla T4 | naive | 62 | measured, 2048^3 |
| Tesla T4 | dp4a | 8,068 | measured, 2048^3 |
| Tesla T4 | tensor_core | 20,363 | measured, 2048^3 |
| Tesla T4 | cublas | 44,031 | measured, 2048^3 |
| Colab CPU | numpy_float32 | 69.2 | measured, 2048^3 |
| 4x4 systolic array (sky130) | pre-layout estimate, 263 MHz | 8.4 | peak, 0.57 mm^2 die |
| 4x4 systolic array (sky130) | post-layout, 100 MHz | 3.2 | peak, 0.57 mm^2 die |

1 multiply-accumulate = 2 ops. The accelerator is a 16-MAC teaching design on an open 130 nm process; the GPU has thousands of MAC units on a 12 nm process, so the gap is expected. The interesting comparison is how each one keeps its MACs fed.
