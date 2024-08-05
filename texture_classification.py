import random
import sys
import numpy as np
import os
from PIL import Image
from KmeansClassifier import KMeansClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from texture_classification_utilities import (show_2d_3d_data,
                                              extract_patches_from_imageset,
                                              extract_clusters_as_arrays_dict_from_indices,
                                              show_patches, compute_gradient_histogram, extract_patches_from_image,
                                              apply_gaussian_blur_to_images)


GLOBAL_PCA = None

'''
Estime le nombre de clusters optimal pour un ensemble de données.
- data: Données d'entraînement.
- range_of_components: Intervalle de valeurs à tester.
- classifier: Classe du classifieur à utiliser.
Retourne : Nombre de clusters optimal.
'''


def estimate_number_of_components(data, range_of_components, classifier):
    silhouette_scores = []
    for n in range_of_components:
        print("Estimating fitting with %d components" % n)
        algorithm = classifier(number_of_components=n)
        algorithm.fit(data)
        labels = algorithm.predict(data)
        print("Computing silhouette score \n")
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    best_fit = np.argmax(silhouette_scores) + range_of_components[0]
    print("best fit with %d components" % best_fit)
    return best_fit


'''
Pré-traite les données d'entrée pour la classification.
- data: Données d'entrée.
- pca_variance: Variance à conserver pour l'ACP.
Retourne : Données pré-traitées.
'''


def texture_preprocessing(data, pca_variance=None):
    print("Preprocessing data ...")
    patches_histograms = np.array(
        [compute_gradient_histogram(patch) for patch in data])
    mean_colors = np.mean(data, axis=(1, 2))
    # Assure que mean_colors est un vecteur colonne.
    if mean_colors.ndim == 1:
        mean_colors = mean_colors[:, np.newaxis]
    # Concaténation des histogrammes et des couleurs moyennes.
    histograms_and_meancolor = np.concatenate(
        (patches_histograms, mean_colors), axis=1)

    # Standardisation et réduction de la dimensionnalité.
    if pca_variance is not None:
        print("Standardization...")
        scaler = StandardScaler()
        histograms_standardized = scaler.fit_transform(patches_histograms)

        print("Dimensionality Reduction ... \n")
        global GLOBAL_PCA
        if GLOBAL_PCA is None:
            GLOBAL_PCA = PCA(pca_variance)
            GLOBAL_PCA.fit(histograms_standardized)
        final_data = GLOBAL_PCA.transform(histograms_standardized)
    else:
        final_data = histograms_and_meancolor

    return final_data


'''
Exécute une démonstration de classification de données 2D.
- filepath: Chemin vers le fichier de données.
'''


def demo(filepath):
    dataset = np.loadtxt(filepath)
    data_train, data_test = train_test_split(
        dataset, test_size=0.5, random_state=42)
    n_components = estimate_number_of_components(
        dataset, range(2, 6), KMeansClassifier)
    kmeans = KMeansClassifier(n_components, 100)
    kmeans.fit(data_train)
    subgroups_indices = kmeans.predict(data_test)
    clusters_arraydict = extract_clusters_as_arrays_dict_from_indices(
        data_test, subgroups_indices)
    show_2d_3d_data(*clusters_arraydict.values(), kmeans.centroids)


'''
Exécute une démonstration de classification de textures.
- filepath: Chemin vers le dossier contenant les images.
- gray: Booléen indiquant si les images doivent être converties en niveaux de gris.
- nb_images: Nombre d'images à traiter. Si None, traite toutes les images du dossier.
- pca_variance: Variance à conserver pour l'ACP.
'''


def demo_texture_kmeans(filepath, gray=False, nb_images=None, pca_variance=None):
    # Fractionnement des données, séparé du prétraitement pour un affichage ultérieur.
    if nb_images is None:
        nb_images = len(os.listdir(filepath))
    base_dataset = extract_patches_from_imageset(
        filepath, (32, 32), nb_images=nb_images)
    base_data_train, base_data_test = train_test_split(
        base_dataset, test_size=0.5, random_state=42)

    if gray:
        data_train = np.mean(base_data_train, axis=3)
        data_test = np.mean(base_data_test, axis=3)
    else:
        data_train = base_data_train
        data_test = base_data_test

    # Données pré-traitées.
    data_train_preprocessed = texture_preprocessing(
        data_train, pca_variance=pca_variance)
    data_test_preprocessed = texture_preprocessing(
        data_test, pca_variance=pca_variance)

    # Entrainement du classifieur.
    kmeans = KMeansClassifier(
        nb_images, max_iterations=200, convergence_threshold=0.01)
    kmeans.fit(data_train_preprocessed)

    # Prédiction du classifieur.
    subgroups_indices = kmeans.predict(data_test_preprocessed)
    classification_arraydict = extract_clusters_as_arrays_dict_from_indices(
        base_data_test, subgroups_indices)

    # Affichage des résultats.
    i = 0
    print("showing samples of randomly selected clusters (max 5)")
    for key in classification_arraydict:
        array = classification_arraydict[key]
        randindex = random.sample(range(len(array)), min(len(array), 100))
        show_patches(array[randindex], (64, 64))
        i += 1
        if i > 5:
            break


'''
Exécute une démonstration de classification d'un patch synthétique.
'''


def demo_generated_patch_classification():
    # Test de classification d'un patch synthétique
    print("calculating the appropriate cluster for textures_data/colored_brodatz_103_synthetic_patch.tif, using train"
          + "data from Colored_Brodatz/subset_5/\n")
    # Fractionnement des données.
    data_train = extract_patches_from_imageset(
        "Colored_Brodatz/subset_5", (32, 32), 5)
    generated_patch = Image.open(
        "textures_data/colored_brodatz_103_synthetic_patch.tif")
    data_test = extract_patches_from_image(generated_patch, (32, 32))

    # Données pré-traitées.
    data_train_preprocessed = texture_preprocessing(data_train)
    data_test_preprocessed = texture_preprocessing(data_test)

    # Entraine le classifieur.
    kmeans = KMeansClassifier(5, max_iterations=200,
                              convergence_threshold=0.01)
    kmeans.fit(data_train_preprocessed)

    # Prédiction du classifieur.
    synthesized_patch_group_index = kmeans.predict(data_test_preprocessed)[0]
    groups_indices = kmeans.predict(data_train_preprocessed)
    classification_arraydict = extract_clusters_as_arrays_dict_from_indices(
        data_train, groups_indices)

    # Affichage des résultats.
    print("showing a sample from the cluster in which the patch has been classified")
    array = classification_arraydict[synthesized_patch_group_index]
    randindex = random.sample(range(len(array)), min(len(array), 100))
    show_patches(array[randindex], (32, 32))


'''
Exécute une démonstration de classification de données floutées.
- filepath: Chemin vers le dossier contenant les images.
'''


def demo_blurred_patches_classification(filepath):
    # Fractionnement des données.
    base_dataset = extract_patches_from_imageset(
        filepath, (32, 32), nb_images=5)
    data_train, base_data_test = train_test_split(
        base_dataset, test_size=0.5, random_state=42)

    data_test = apply_gaussian_blur_to_images(base_data_test)

    # Donnéees pré-traitées.
    data_train_preprocessed = texture_preprocessing(data_train)
    data_test_preprocessed = texture_preprocessing(data_test)

    # Entraine le classifieur.
    kmeans = KMeansClassifier(5, max_iterations=200,
                              convergence_threshold=0.01)
    kmeans.fit(data_train_preprocessed)

    # Prédiction du classifieur.
    subgroups_indices = kmeans.predict(data_test_preprocessed)
    classification_arraydict = extract_clusters_as_arrays_dict_from_indices(
        data_test, subgroups_indices)

    # Affichage des résultats.
    i = 0
    print("showing samples of each calculated clusters")
    for key in classification_arraydict:
        array = classification_arraydict[key]
        randindex = random.sample(range(len(array)), min(len(array), 100))
        show_patches(array[randindex], (64, 64))
        i += 1
        if i > 5:
            break


'''
Exécute une démonstration en fonction des arguments passés au script.
'''
if __name__ == "__main__":
    if len(sys.argv) == 2:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if sys.argv[1] == "demo_2d_kmeans":
            file_path = os.path.join(script_dir, "classif_data/gmm2d.asc")
            demo(file_path)

        elif sys.argv[1] == "demo_3d_kmeans":
            file_path = os.path.join(script_dir, "classif_data/gmm3d.asc")
            demo(file_path)

        elif sys.argv[1] == "demo_patch_extraction":
            print("cutting one example image in 64x64 patches ... \n")
            patch_size = (64, 64)
            dir_path = os.path.join(script_dir, "Colored_Brodatz")
            patches = extract_patches_from_imageset(
                dir_path, patch_size, nb_images=1)
            show_patches(patches, patch_size)

        elif sys.argv[1] == "demo_texture_kmeans_gray":
            print("testing k-means classification on 5 sample images ...")
            dir_path = os.path.join(script_dir, "Colored_Brodatz")
            demo_texture_kmeans(dir_path, gray=True, nb_images=5)

        elif sys.argv[1] == "demo_texture_kmeans_color":
            print("testing k-means classification on 5 sample images ...")
            dir_path = os.path.join(script_dir, "Colored_Brodatz")
            demo_texture_kmeans(dir_path, gray=False, nb_images=5)

        elif sys.argv[1] == "demo_synthesized_patch_classification":
            demo_generated_patch_classification()

        elif sys.argv[1] == "demo_blur_classification":
            print("classifying image patches after blurring them")
            dir_path = os.path.join(script_dir, "Colored_Brodatz")
            demo_blurred_patches_classification(dir_path)

        else:
            print("Usage : python3.8 texture_classification.py <demo_name> \n"
                  "Available demos : demo_2d_kmeans, demo_3d_kmeans, demo_texture_kmeans_gray, "
                  + "demo_texture_kmeans_color, demo_synthesized_patch_classification, demo_blur_classification.\n")
            exit(1)

    else:
        print("Usage : python3.8 texture_classification.py <demo_name> \n"
              "Available demos : demo_2d_kmeans, demo_3d_kmeans, demo_texture_kmeans_gray, "
              + "demo_texture_kmeans_color, demo_synthesized_patch_classification, demo_blur_classification.\n")
        exit(1)
