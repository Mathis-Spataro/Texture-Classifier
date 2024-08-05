import numpy as np
import random

'''
Classe implémentant l'algorithme de classification K-Means.
'''


class KMeansClassifier:
    '''
    Initialisation des paramètres de l'algorithme.
    - number_of_components: Nombre de clusters à créer.
    - max_iterations: Nombre maximum d'itérations à effectuer.
    - convergence_threshold: Seuil de convergence de l'algorithme.
    '''

    def __init__(self, number_of_components, max_iterations=100, convergence_threshold=0.05):
        self.n_components = number_of_components
        self.n_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.current_iteration = 0
        self.centroids = np.empty(0)
        self.last_iteration_centroids = np.empty(number_of_components)
        # Contient l'indice du centroïde assigné à chaque point.
        self.data_centroid_association = np.empty(0)

    '''
    Initialisation des structures de données et des centroïdes.
    - data: Données d'entraînement.
    '''

    def init_fit(self, data):
        # Initialisation des structures de données
        self.data_centroid_association = np.empty(data.shape[0])
        self.centroids = np.empty((self.n_components,) + data.shape[1:])

        # Initialisation aléatoire des centroïdes
        random_indexes = random.sample(range(len(data)), self.n_components)
        self.centroids = data[random_indexes]

        # Initialisation avec des valeurs garanties de ne pas dépasser le seuil de convergence
        self.last_iteration_centroids = -self.centroids
        self.current_iteration = 0

    '''
    Vérification de la convergence de l'algorithme.
    Retourne : True si la convergence est atteinte, False sinon.
    '''

    def check_convergence_is_finished(self):
        # Calcul de la distance euclidienne entre les positions actuelles et précédentes des centroïdes
        centroids_distances_to_last_positions = np.linalg.norm(
            self.centroids - self.last_iteration_centroids, axis=1)

        # Calcul du pourcentage de mouvement des centroïdes
        centroids_movement_amount = np.sum(
            centroids_distances_to_last_positions) / np.sum(abs(self.last_iteration_centroids), axis=1)

        # Vérification de la convergence en fonction des itérations ou du mouvement des centroïdes
        if self.current_iteration >= self.n_iterations or centroids_movement_amount.all() < self.convergence_threshold:
            return True
        return False

    '''
    Assignation des points aux clusters.
    - data: Données d'entraînement.
    '''

    def assignment_step(self, data):
        # Assignation de chaque point au centroïde le plus proche
        for index, item in enumerate(data):
            distances = np.linalg.norm(self.centroids - item, axis=1)
            self.data_centroid_association[index] = np.argmin(distances)

    '''
    Mise à jour des positions des centroïdes en fonction des points associés.
    - data: Données d'entraînement.
    '''

    def update_step(self, data):
        self.last_iteration_centroids = self.centroids.copy()
        # Mise à jour des positions des centroïdes en fonction des points associés à chacun d'eux (moyenne).
        for c in range(len(self.centroids)):
            centroid_associated_indexes = [index for index, value in enumerate(self.data_centroid_association)
                                           if value == c]
            self.centroids[c] = np.mean(
                data[centroid_associated_indexes], axis=0)

    '''
    Exécution de l'algorithme.
    - data: Données d'entraînement.
    '''

    def fit(self, data):
        self.init_fit(data)
        # Tant que la convergence n'est pas atteinte, on répète les étapes d'assignation et de mise à jour.
        while not self.check_convergence_is_finished():
            print("iteration : %d" % self.current_iteration)
            self.current_iteration += 1
            self.assignment_step(data)
            self.update_step(data)

    '''
    Prédiction des associations de clusters pour de nouveaux points.
    - data: Données d'entraînement.
    Retourne : Les indices des clusters associés à chaque point.
    '''

    def predict(self, data):
        cluster_association = np.empty(data.shape[0])
        # Assignation de chaque point au centroïde le plus proche.
        for index, data_point in enumerate(data):
            distances = np.linalg.norm(self.centroids - data_point, axis=1)
            cluster_association[index] = np.argmin(distances)
        return cluster_association
