# [R2D2](https://arxiv.org/abs/2403.05452) Claims

They start with ground truth images from public datasets.
Then they simulate the forward pass to get simulated visibility data.
- Section 3.2 Ground truth database, pg 5:
```
Key to the training of our deep-learning model is a
large database of ground-truth images with a wide variety of features and dynamic ranges, and free from noise
and artifact structure. In the absence of a large physical
RI database readily providing these characteristics, we
build a database of curated ground-truth images from
real low-dynamic range astronomical and medical images, sourced as follows. Radio astronomy images are
gathered from the National Radio Astronomy Observatory (NRAO) Archives, and LOFAR surveys, namely,
LOFAR HBA Virgo cluster survey (Edler et al. 2023)
and LoTSS-DR2 survey (Shimwell et al. 2022). Optical astronomy images are gathered from the National
Optical-Infrared Astronomy Research Laboratory. Medical images are selected from the NYU fastMRI Initiative Database (Zbontar et al. 2018; Knoll et al. 2020).
Training using a curated database originating from other
modalities and applications has shown to be effective for
RI imaging (Terris et al. 2022, 2023).
Ground-truth images of size N = 512 × 512, are generated using the pre-processing procedure proposed in
Terris et al. (2023). More precisely, various operations
including concatenation, rotation, translation, and edge
smoothing, are applied specifically to the medical images to deconstruct their anatomical features. Denoising is applied to all images, of both medical and astronomical origins, to eliminate artifacts and noise, using
a denoising DNN (Zhang et al. 2023) in combination
with soft-thresholding operations. Additionally, a pixelwise exponentiation transform can be applied to the curated ground-truth images to emulate the characteristic
high dynamic range of radio images (Terris et al. 2022).
Examples of raw low-dynamic range images and their
corresponding denoised and exponentiated ground-truth
images are shown in Figure 3.
```

They train specifically for the VLA.
```
Training was conducted on Cirrus, a UK Tier 2 highperformance computing (HPC) service, equipped with
both CPU and GPU compute nodes. The CPU nodes
are composed of dual Intel 18-core Xeon E5-2695 processors with 256 GB of memory each. The GPU nodes
are composed of two 20-core Intel Xeon Gold 6148 processors, four NVIDIA Tesla V100-SXM2-16GB GPUs,
and 384 GB of DRAM memory. Computation of the
dirty images and updates of the residual dirty images
were run on the CPU nodes. DNNs’ training relied on
the PyTorch library in Python (Paszke et al. 2019), and
was performed on the GPU nodes.
```
