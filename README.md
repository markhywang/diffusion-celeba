# Diffusion Models

Diffusion Models are commonly considered stronger alternatives compared to Generative Adversarial Networks (GANs) for performing generative AI tasks like text-to-image or image-to-image generation.
<br><br>
Diffusion Models learns spatial patterns and thus gains its ability to accurately generate images through repetitively adding noise to the image until it reaches an Isotropic Gaussian Distribution. Afterwards, it uses a U-Net to predict the noise added in each timestep, which is then used to de-noise the noisy image into a proper image again.
<br><br>
The equation for denoising at a timestep t from the image at the timestep t-1 is given as follows:
![alt text](https://github.com/markhywang/diffusion-celeba/blob/main/assets/forward-equation.png)
<br><br>
Note that this equation can be further optimized during pre-computation by taking cumulative products of consecutive beta values, which are denoted by alpha values in the implementation.
<br><br>
When de-noising the image, a U-Net is used for image segmentation to predict the noise added in each timestep, which is then subtracting from the image at thet timestep to get a less noisy image.
<br><br>
The loss function for backward diffusion is as follows:
![alt text](https://github.com/markhywang/diffusion-celeba/blob/main/assets/backward-equation.png)
