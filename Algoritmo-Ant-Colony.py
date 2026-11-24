"""
Algoritmo de Colonia de Hormigas (Ant Colony Optimization, ACO)
Aplicado al Problema del Agente Viajero (TSP)

Materia: Algoritmos Bioinspirados
Alumno: Dante Yael Macías Rentería

Idea bioinspirada:
  Las hormigas reales encuentran el camino más corto entre el nido y la comida
  dejando feromona en el suelo. Los caminos cortos se recorren más veces en el
  mismo tiempo, acumulan más feromona y atraen a más hormigas. La feromona
  también se evapora, lo que evita quedarse atrapado en caminos malos.

En el algoritmo:
  - Cada hormiga construye un recorrido visitando todas las ciudades una vez.
  - La probabilidad de ir de la ciudad i a la j depende de:
        tau(i,j)^alpha  -> cantidad de feromona en la arista
        eta(i,j)^beta   -> visibilidad heurística = 1 / distancia(i,j)
  - Al final de cada iteración:
        1) Se evapora la feromona:      tau = (1 - rho) * tau
        2) Cada hormiga deposita:       tau(i,j) += Q / L_k
           (L_k = longitud del recorrido de la hormiga k)
"""

import numpy as np
import matplotlib.pyplot as plt


class ColoniaHormigas:
    def __init__(self, distancias, n_hormigas=20, n_iteraciones=100,
                 alpha=1.0, beta=5.0, rho=0.5, Q=100, semilla=None):
        """
        distancias    : matriz NxN con las distancias entre ciudades
        n_hormigas    : número de hormigas por iteración
        n_iteraciones : número de iteraciones (generaciones)
        alpha         : importancia de la feromona
        beta          : importancia de la heurística (1/distancia)
        rho           : tasa de evaporación (0 a 1)
        Q             : constante de depósito de feromona
        """
        self.dist = distancias
        self.n = len(distancias)
        self.n_hormigas = n_hormigas
        self.n_iteraciones = n_iteraciones
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.rng = np.random.default_rng(semilla)

        # Feromona inicial: un valor pequeño e igual en todas las aristas
        self.feromona = np.ones((self.n, self.n))

        # Visibilidad: 1/distancia (se evita dividir entre cero en la diagonal)
        with np.errstate(divide="ignore"):
            self.visibilidad = 1.0 / distancias
        np.fill_diagonal(self.visibilidad, 0)

    def construir_recorrido(self):
        """Una hormiga construye un recorrido completo."""
        inicio = self.rng.integers(self.n)
        recorrido = [inicio]
        visitadas = {inicio}

        while len(recorrido) < self.n:
            actual = recorrido[-1]
            probabilidades = self.calcular_probabilidades(actual, visitadas)
            siguiente = self.rng.choice(self.n, p=probabilidades)
            recorrido.append(siguiente)
            visitadas.add(siguiente)

        return recorrido

    def calcular_probabilidades(self, actual, visitadas):
        """Regla de transición proporcional:
           P(i,j) = [tau^alpha * eta^beta] / suma de lo mismo para las no visitadas
        """
        tau = self.feromona[actual] ** self.alpha
        eta = self.visibilidad[actual] ** self.beta
        atractivo = tau * eta

        # Las ciudades ya visitadas tienen probabilidad 0
        atractivo[list(visitadas)] = 0
        return atractivo / atractivo.sum()

    def longitud(self, recorrido):
        """Distancia total del recorrido (regresando a la ciudad inicial)."""
        total = 0
        for k in range(self.n):
            i = recorrido[k]
            j = recorrido[(k + 1) % self.n]
            total += self.dist[i][j]
        return total

    def actualizar_feromona(self, recorridos, longitudes):
        # 1) Evaporación
        self.feromona *= (1 - self.rho)

        # 2) Depósito: cada hormiga deja Q/L en las aristas que usó
        for recorrido, L in zip(recorridos, longitudes):
            deposito = self.Q / L
            for k in range(self.n):
                i = recorrido[k]
                j = recorrido[(k + 1) % self.n]
                self.feromona[i][j] += deposito
                self.feromona[j][i] += deposito  # grafo no dirigido

    def ejecutar(self, mostrar=True):
        mejor_recorrido = None
        mejor_longitud = float("inf")
        historial = []

        for it in range(self.n_iteraciones):
            recorridos = [self.construir_recorrido() for _ in range(self.n_hormigas)]
            longitudes = [self.longitud(r) for r in recorridos]

            # Guardar el mejor encontrado hasta ahora
            idx = int(np.argmin(longitudes))
            if longitudes[idx] < mejor_longitud:
                mejor_longitud = longitudes[idx]
                mejor_recorrido = recorridos[idx]

            self.actualizar_feromona(recorridos, longitudes)
            historial.append(mejor_longitud)

            if mostrar and (it % 10 == 0 or it == self.n_iteraciones - 1):
                print(f"Iteración {it + 1:3d} | Mejor distancia: {mejor_longitud:.2f}")

        return mejor_recorrido, mejor_longitud, historial


def matriz_distancias(coordenadas):
    """Distancia euclidiana entre todas las ciudades."""
    diferencias = coordenadas[:, None, :] - coordenadas[None, :, :]
    return np.sqrt((diferencias ** 2).sum(axis=2))


def graficar(coordenadas, recorrido, historial):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Mejor ruta encontrada
    ruta = recorrido + [recorrido[0]]
    ax1.plot(coordenadas[ruta, 0], coordenadas[ruta, 1], "o-", color="tab:blue")
    ax1.plot(coordenadas[ruta[0], 0], coordenadas[ruta[0], 1], "s", color="tab:red",
             markersize=10, label="Inicio")
    for i, (x, y) in enumerate(coordenadas):
        ax1.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5))
    ax1.set_title(f"Mejor ruta encontrada (distancia = {historial[-1]:.2f})")
    ax1.legend()

    # Convergencia
    ax2.plot(range(1, len(historial) + 1), historial, color="tab:green")
    ax2.set_title("Convergencia del algoritmo")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Mejor distancia")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultado_aco.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    # Generamos 20 ciudades aleatorias en un plano de 100x100
    rng = np.random.default_rng(42)
    n_ciudades = 20
    coordenadas = rng.uniform(0, 100, size=(n_ciudades, 2))
    distancias = matriz_distancias(coordenadas)

    aco = ColoniaHormigas(
        distancias,
        n_hormigas=20,
        n_iteraciones=100,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=100,
        semilla=1,
    )

    mejor_recorrido, mejor_longitud, historial = aco.ejecutar()

    print("\n=== RESULTADO ===")
    print("Mejor recorrido:", " -> ".join(map(str, mejor_recorrido + [mejor_recorrido[0]])))
    print(f"Distancia total: {mejor_longitud:.2f}")

    graficar(coordenadas, mejor_recorrido, historial)
