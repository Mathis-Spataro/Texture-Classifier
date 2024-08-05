import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

"""----------------  Data management ----------------"""

'''
Extrait les patches d'une image.
- image: Image à traiter.
- patch_size: Taille des patches à extraire.
Retourne : Tableau de patches.
'''


def extract_patches_from_image(image, patch_size):
    width, height = image.size
    patch_width, patch_height = patch_size
    patches = []
    for x in range(0, width, patch_width):
        for y in range(0, height, patch_height):
            patch = image.crop((x, y, x + patch_width, y + patch_height))
            patch_array = np.array(patch)
            patches.append(patch_array)
    return np.array(patches)


'''
Extrait les patches de chaque image .tif du dossier data_folder. 
- data_folder: Chemin du dossier contenant les images.
- patch_size = Tuple d'entiers.
- nb_images: Nombre d'images à traiter. Si None, traite toutes les images du dossier.
Retourne : Tableau de patches de toutes les images.
'''


def extract_patches_from_imageset(data_folder, patch_size, nb_images=None):
    patches = []
    for index, filename in enumerate(os.listdir(data_folder)):
        if filename.endswith(".tif"):
            image_path = os.path.join(data_folder, filename)
            image_patches = extract_patches_from_image(
                Image.open(image_path), patch_size)
            patches.append(image_patches)
        if nb_images is not None and index >= nb_images - 1:
            break
    patches_numpy = np.vstack(patches)
    return patches_numpy


'''
Extrait les clusters d'un ensemble de données en fonction des indices de cluster.
- data: Données d'entraînement.
- indices: Indices des données à extraire.
Retourne : Tableau de clusters.
'''


def extract_clusters_as_arrays_dict_from_indices(data, indices):
    arrays_dict = {}

    for data_index, cluster_index in enumerate(indices):
        if cluster_index not in arrays_dict:
            arrays_dict[cluster_index] = []
        arrays_dict[cluster_index].append(data[data_index])

    # Convertis le dictionnaire en un dictionnaire de tableaux numpy.
    for key, cluster_array in arrays_dict.items():
        arrays_dict[key] = np.array(cluster_array)

    return arrays_dict


'''
Calcule l'histogramme des gradients d'une image.
- image: Image à traiter.
Retourne : Tableau numpy de l'histogramme des gradients.
'''


def compute_gradient_histogram(image):
    if image.shape[-1] == 3:
        hist = np.empty(36 * image.shape[2])
        for i in range(3):
            # Calcule les gradients en utilisant les filtres Sobel.
            grad_x = cv2.Sobel(image[:, :, i], cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image[:, :, i], cv2.CV_64F, 0, 1, ksize=3)
            # Calcule la magnitude et la phase (angle) des gradients.
            magnitude, angle = cv2.cartToPolar(grad_x, grad_y)

            # Compile l'histogramme des angles des gradients.
            tmphist, bins = np.histogram(angle, bins=36, range=(0, 2 * np.pi))
            hist[i * 36:(i + 1) * 36] = tmphist
    else:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y)
        hist, bins = np.histogram(angle, bins=36, range=(0, 2 * np.pi))

    return np.array(hist)


'''
Applique un flou gaussien à un tableau d'images.
- images: Tableau d'images.
- kernel_size: Taille du kernel du flou gaussien.
Retourne : Tableau d'images floutées avec le même ordre que l'entrée.
'''


def apply_gaussian_blur_to_images(images, kernel_size=(15, 15)):
    blurred_images = []
    for image in images:
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        blurred_images.append(blurred_image)
    return blurred_images


"""----------------  Data visualization ----------------"""

'''
Affiche les données en 2D ou 3D.
- data_arrays: Tableaux de données à afficher.
'''


def show_2d_3d_data(*data_arrays):
    if len(data_arrays[0][0]) == 3:
        show_3d_data(*data_arrays)
    else:
        show_2d_data(*data_arrays)


'''
Affiche les données en 3D.
- data_arrays: Tableaux de données à afficher.
Précondition : Le dernier tableau doit être les centroïdes.
'''


def show_3d_data(*data_arrays):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    num_arrays = len(data_arrays)

    for i, data in enumerate(data_arrays):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        label = f'Cluster {i + 1}'  # Label pour la légende.
        if i == num_arrays - 1:
            label = f'Centroids'

        # Crée un scatter plot avec les données du tableau.
        ax.scatter(x, y, z, c=color, marker=marker, label=label)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('Scatter Plot of 3D Data Points')
    plt.legend()
    plt.show()


'''
Affiche les données en 2D.
- data_arrays: Tableaux de données à afficher.
Précondition : Le dernier tableau doit être les centroïdes.
'''


def show_2d_data(*data_arrays):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    num_arrays = len(data_arrays)

    for i, data in enumerate(data_arrays):
        x = data[:, 0]
        y = data[:, 1]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        label = f'Cluster {i + 1}'
        if i == num_arrays - 1:
            label = f'Centroids'

        # Plot the data array with the specified color and marker
        plt.scatter(x, y, c=color, marker=marker, label=label)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Data Points')
    plt.legend()
    plt.show()


'''
Affiche les patches.
- patches: Patches à afficher.
- patch_size: Taille des patches.
- linearized: Si True, les patches sont linéarisés.
'''


def show_patches(patches, patch_size, linearized=False):
    num_patches = patches.shape[0]
    num_rows = int(np.sqrt(num_patches))
    num_cols = int(np.ceil(num_patches / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i, ax in enumerate(axes.ravel()):
        if i < num_patches:
            if linearized:
                patch = patches[i].reshape(patch_size[0], patch_size[1], 3)
            else:
                patch = patches[i]
            ax.imshow(patch)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()
