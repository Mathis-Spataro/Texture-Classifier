# Texture generation and classification software - Autumn 2023

This software was an assignment project for our Statistical and Frequential Models for Image Analysis class during my second year of Master's degree.

There was no starting code.

## work done

We've implemented the following features :
- Texture extension by generating new pixels from the original texture.
- K-means algorithm
- Command line interface for using the previous features

## Implemented algorithm details
This section is a short version of the french report available in the root folder of the repository.

### Texture generation

To extend our textures by generating pixels, we rely on three main components :
- A patch extractor divides the textures from the dataset into square patches.
- A gaussian mask weights the pixels of each patch, according more value to the pixels in the middle of them.
- A squared distance sum compares the environment of a pixel to generate to each patch, resulting in a "closeness score" for each of them.
Once this is done we select the closest patch and apply the color in the middle of it to the pixel to generate.
The squared distance sum is repeated for each pixel to generate until the image is filled up.

### Texture classification

For our K-means algorithm, we used the patch extractor to inflate our dataset. We trained the model on half of the data and tested it on the other.

A K-means algorithm relies on clusters and centroids. The data pieces are linked to the closest centroid, and form a cluster.
During training, we repeatedly recompute the position of the centroids as the mean of all the data pieces forming its associated cluster.
By doing so, while data pieces are unchanged, some of the data will change clusters, and the centroids converge to a final position where they don't move anymore.

Once training is done, classification of a new data piece is done by computing its closest centroid.

## Results

### Texture synthesis

Here is an example of texture we've extended using our algorithm. (original on the left)

<div style="display: flex; justify-content: space-between;">
  <img src="results/texture synthesis/synthesis_texture0_64x64.png" alt="Original Texture" style="width: 30%; margin-right: 10px;">
  <img src="results/texture synthesis/synthesis_texture0_128x128_(2).png" alt="Extended Texture" style="width: 30%;">
</div>

### Texture classification

Here is a series of 3D points we've classified using our K-means algorithm :

<img src="results/points classification/classification_k-means_4_components_3D.png" alt="3D points classification" style="width: 75%;">

Here is a group of textures, divided in patches, we've classified (each full image represents a resulting cluster) :
<div style="display: flex; justify-content: space-between;">
  <img src="results/texture classification/colored brodatz classification/5_images_classification_cluster_1_sample.png" alt="Classification Texture Image 1" style="width: 30%; margin-right: 10px;">
  <img src="results/texture classification/colored brodatz classification/5_images_classification_cluster_2_sample.png" alt="Classification Texture Image 2" style="width: 30%; margin-right: 10px;">
  <img src="results/texture classification/colored brodatz classification/5_images_classification_cluster_3_sample.png" alt="Classification Texture Image 3" style="width: 30%;">
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="results/texture classification/colored brodatz classification/5_images_classification_cluster_4_sample.png" alt="Classification Texture Image 4" style="width: 30%; margin-right: 10px;">
  <img src="results/texture classification/colored brodatz classification/5_images_classification_cluster_5_sample.png" alt="Classification Texture Image 5" style="width: 30%;">
</div>

Finally, we've attempted to classify a generated patch of data from one of the dataset's texture, and managed to associate it with similar patches :

<div style="display: flex; justify-content: space-between;">
  <img src="results/texture classification/synthetic patch classification/synthetic_patch_to_classify.png" alt="Generated Patch to Classify" style="width: 20%; margin-right: 10px;">
  <img src="results/texture classification/synthetic patch classification/synthesized_patch_cluster_classification.png" alt="Associated Cluster to Generated Image" style="width: 50%;">
</div>





