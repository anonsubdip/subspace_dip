# subspace_dip_learning
Here, we include the PyTorch implementation for the work submitted to ICML 2023 (https://openreview.net/pdf?id=5d53OeOZHE)

# Novel Experimental Investigation (Rebuttal)
## Showcasing the Sub-DIP on Image Restoration Tasks using Natural Images

**Step 1: Identifying the Sparse Subspace** 

We use the [PASCAL VOC](https://paperswithcode.github.io/torchbench/pascalvoc/) natural image dataset (1472 RGB natural images) to pre-train a shallow U-Net with 64 channels and five scales for 2000 epochs, while retaining 5000 checkpoints, equally-spaced throughout the optimisation trajectory. We downsize the pre-training image size to $128\times 128$ through random cropping to speed-up pre-training. We solve either a denoising or a deblurring task. For denoising, 25% Gaussian noise is added to the ground truth images. For deblurring we use a Gaussian blurring kernel with standard deviation of 1.6 pixels and 5% Gaussian noise. After pre-training, we extract a ($d_{\rm sub} = 4k$) subspace with a sparsity level of $d_{\textrm{lev}} / d_{\theta}$ of 0.75.

**Step 2: Natural Images Restoration (Denoising and Deblurring)**

We report two additional experimental figures on a denosing and deblurring tasks, where the standard DIP and Sub-DIP (NGD) are compared on three widely used [RGB images](https://sipi.usc.edu/database/database.php?volume=misc&image=3#top), namely Airplane F16, House, and Lena. Note that for both restoration task, the image resolution used is $256\times 256$, four times larger than the size used at pre-training. 

For denoising, at test time 10% white noise is removed from the images below.

![natural_images_identity](https://user-images.githubusercontent.com/123627605/226110875-9fa2ed1c-b6fa-4281-bb04-e54455dad9f7.png)

For deblurring, at test time a Gaussian blur with standard deviation of 1.6 pixels is used along with 5% Gaussian noise.

![natural_images_blurring](https://user-images.githubusercontent.com/123627605/226110886-dab86755-7f1f-49c7-a5fa-86df359ceafb.png)
