import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

# Paso 1: Utilizar los mismos datos estandarizados
X = df.drop('etiqueta', axis=1).values  # Usamos los mismos datos que con K-means

# Parámetros de K-medoides
n_clusters = 40  # Número de clusters
random_state = 123
n_init = 10  # Número de inicializaciones aleatorias para elegir la mejor configuración
tolerancia = 0.001
max_iter = 300

np.random.seed(random_state)

# Lista de valores de p para la distancia de Minkowski
p_values = [1, 2, 3, 4, 5]  # Puedes agregar más valores de p para experimentar
resultados_precision = []

# Iterar sobre diferentes valores de p
for p in p_values:
    print(f"\nEvaluando con p = {p}")

    # Mejor configuración inicial
    mejor_medoids = None
    mejor_costo = np.inf
    mejor_etiquetas = None

    # Iteramos por el número de inicializaciones aleatorias (n_init)
    for init_no in range(n_init):
        print(f"Inicialización {init_no + 1} de {n_init}")
        
        # Inicializamos medoids aleatoriamente
        medoid_idxs = np.random.choice(range(X.shape[0]), n_clusters, replace=False)
        medoids = X[medoid_idxs]
        
        iteracion = 0
        cambio_costo = np.inf
        costo_actual = np.inf
        
        # Iteramos hasta la convergencia o hasta alcanzar el número máximo de iteraciones
        while iteracion < max_iter and cambio_costo > tolerancia:
            iteracion += 1
            
            # Paso 3: Asignar cada punto al medoid más cercano utilizando distancia de Minkowski
            distancias = cdist(X, medoids, metric='minkowski', p=p)
            etiquetas = np.argmin(distancias, axis=1)
            
            # Paso 4: Calcular el costo
            costo_anterior = costo_actual
            costo_actual = np.sum(np.min(distancias, axis=1))
            cambio_costo = costo_anterior - costo_actual
            
            # Actualizar los medoids
            nuevos_medoids = np.copy(medoids)
            for k in range(n_clusters):
                cluster_points = X[etiquetas == k]
                if len(cluster_points) > 0:
                    minkowski_distances = cdist(cluster_points, cluster_points, metric='minkowski', p=p)
                    medoid_idx = np.argmin(np.sum(minkowski_distances, axis=1))
                    nuevos_medoids[k] = cluster_points[medoid_idx]

            # Verificar convergencia basada en el cambio de medoids
            if np.all(nuevos_medoids == medoids):
                print(f"Convergencia alcanzada en la iteración {iteracion}")
                break

            medoids = nuevos_medoids
        
        # Guardar la mejor configuración de medoids
        if costo_actual < mejor_costo:
            mejor_costo = costo_actual
            mejor_medoids = medoids
            mejor_etiquetas = etiquetas

    # Utilizar la mejor configuración encontrada
    medoids = mejor_medoids
    etiquetas = mejor_etiquetas

    # Paso 5: Evaluar la precisión del algoritmo para agrupar las imágenes
    etiquetas_pred = np.zeros_like(etiquetas)

    for i in np.unique(etiquetas):
        indices_cluster = (etiquetas == i)
        etiquetas_cluster = df['etiqueta'][indices_cluster]
        counter = Counter(etiquetas_cluster)
        etiqueta_cluster = counter.most_common(1)[0][0]
        etiquetas_pred[indices_cluster] = etiqueta_cluster

    accuracy = accuracy_score(df['etiqueta'], etiquetas_pred)
    print(f"Precisión con K-medoides (Distancia de Minkowski, p={p}): {accuracy:.4f}")

    # Almacenar la precisión para el valor actual de p
    resultados_precision.append((p, accuracy))

    # Visualización del histograma
    precision_por_individuo_kmedoids = {}
    for individuo in np.unique(df['etiqueta']):
        indices_individuo = (df['etiqueta'] == individuo)
        etiquetas_pred_individuo = etiquetas_pred[indices_individuo]
        etiquetas_verdaderas_individuo = df['etiqueta'][indices_individuo]
        precision_individuo = accuracy_score(etiquetas_verdaderas_individuo, etiquetas_pred_individuo)
        precision_por_individuo_kmedoids[individuo] = precision_individuo

    precisiones_kmedoids = list(precision_por_individuo_kmedoids.values())
    plt.figure(figsize=(10, 6))
    plt.hist(precisiones_kmedoids, bins=10, color='blue', alpha=0.7)
    plt.title(f'Histograma de Precisión por Individuo con K-medoides (Distancia de Minkowski, p={p})')
    plt.xlabel('Precisión')
    plt.ylabel('Número de Individuos')
    plt.grid(True)
    plt.show()

# Visualizar la precisión para todos los valores de p
p_vals, precisiones = zip(*resultados_precision)
plt.figure(figsize=(8, 6))
plt.plot(p_vals, precisiones, marker='o', linestyle='-')
plt.title('Precisión del Modelo con Diferentes Valores de p para la Distancia de Minkowski')
plt.xlabel('Valor de p')
plt.ylabel('Precisión')
plt.grid(True)
plt.show()