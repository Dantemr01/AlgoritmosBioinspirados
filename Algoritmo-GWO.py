"""
Optimizador del Lobo Gris (Grey Wolf Optimizer, GWO)
Aplicado a la minimización de funciones de prueba (Rastrigin)
Basado en: Mirjalili, Mirjalili y Lewis (2014)

Materia: Algoritmos Bioinspirados
Alumno: Dante Yael Macías Rentería

Idea bioinspirada:
  Imita la jerarquía social y la forma de cazar de los lobos grises.
  La manada tiene cuatro rangos:
    - Alfa  (α): el líder, la mejor solución encontrada
    - Beta  (β): el segundo mejor, ayuda al alfa
    - Delta (δ): el tercer mejor
    - Omega (ω): el resto de la manada, que sigue a los tres líderes
  La caza tiene tres fases: buscar la presa, rodearla y atacarla.
  Como no se sabe dónde está la presa (el óptimo), se supone que alfa, beta
  y delta tienen la mejor idea de su ubicación, y los demás lobos se mueven
  guiados por ellos.

En el algoritmo, para cada lobo X y cada líder (α, β, δ):

    A = 2·a·r1 - a          C = 2·r2          (r1, r2 aleatorios en [0, 1])
    D = |C·X_lider - X|                       -> distancia al líder
    X_i = X_lider - A·D                       -> posición sugerida por ese líder

  La nueva posición del lobo es el promedio de las tres sugerencias:

    X(t+1) = (X1 + X2 + X3) / 3

  El parámetro "a" baja linealmente de 2 a 0 durante las iteraciones:
    - Si |A| > 1, el lobo se aleja de la presa  -> EXPLORACIÓN (búsqueda)
    - Si |A| < 1, el lobo se acerca a la presa  -> EXPLOTACIÓN (ataque)
"""

import numpy as np
import matplotlib.pyplot as plt

def esfera(x):
    """Función esfera: fácil, un solo mínimo."""
    return np.sum(x ** 2, axis=-1)


def rastrigin(x):
    """Función de Rastrigin: muchos mínimos locales, difícil de optimizar."""
    n = x.shape[-1]
    return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=-1)


class LoboGris:
    def __init__(self, funcion, dimensiones, limites, n_lobos=30,
                 n_iteraciones=100, semilla=None):
        """
        funcion       : función objetivo a minimizar
        dimensiones   : número de variables del problema
        limites       : (mínimo, máximo) del espacio de búsqueda
        n_lobos       : tamaño de la manada
        n_iteraciones : número de iteraciones
        """
        self.f = funcion
        self.d = dimensiones
        self.lim_inf, self.lim_sup = limites
        self.n_lobos = n_lobos
        self.n_iteraciones = n_iteraciones
        self.rng = np.random.default_rng(semilla)

        # Posiciones iniciales aleatorias de la manada
        self.X = self.rng.uniform(self.lim_inf, self.lim_sup,
                                  size=(n_lobos, dimensiones))

        # Líderes: posición y valor (se inicializan "infinitamente malos")
        self.alfa, self.alfa_valor = np.zeros(dimensiones), float("inf")
        self.beta, self.beta_valor = np.zeros(dimensiones), float("inf")
        self.delta, self.delta_valor = np.zeros(dimensiones), float("inf")

    def actualizar_lideres(self):
        """Evalúa la manada y actualiza alfa, beta y delta."""
        valores = self.f(self.X)

        for i in range(self.n_lobos):
            v = valores[i]
            if v < self.alfa_valor:
                # El nuevo lobo es alfa; los anteriores bajan de rango
                self.delta, self.delta_valor = self.beta.copy(), self.beta_valor
                self.beta, self.beta_valor = self.alfa.copy(), self.alfa_valor
                self.alfa, self.alfa_valor = self.X[i].copy(), v
            elif v < self.beta_valor:
                self.delta, self.delta_valor = self.beta.copy(), self.beta_valor
                self.beta, self.beta_valor = self.X[i].copy(), v
            elif v < self.delta_valor:
                self.delta, self.delta_valor = self.X[i].copy(), v

    def movimiento_hacia(self, lider, a):
        """Posición que le sugiere un líder a cada lobo (rodear a la presa)."""
        r1 = self.rng.random((self.n_lobos, self.d))
        r2 = self.rng.random((self.n_lobos, self.d))
        A = 2 * a * r1 - a
        C = 2 * r2
        D = np.abs(C * lider - self.X)
        return lider - A * D

    def actualizar_posiciones(self, a):
        X1 = self.movimiento_hacia(self.alfa, a)
        X2 = self.movimiento_hacia(self.beta, a)
        X3 = self.movimiento_hacia(self.delta, a)

        # La nueva posición es el promedio de lo que indican los tres líderes
        self.X = (X1 + X2 + X3) / 3
        # Mantener a los lobos dentro del espacio de búsqueda
        self.X = np.clip(self.X, self.lim_inf, self.lim_sup)

    def ejecutar(self, mostrar=True):
        historial = []
        posiciones = [self.X.copy()]  # para graficar cómo se mueve la manada

        for it in range(self.n_iteraciones):
            self.actualizar_lideres()

            # "a" disminuye linealmente de 2 a 0
            a = 2 - 2 * it / self.n_iteraciones

            self.actualizar_posiciones(a)
            posiciones.append(self.X.copy())
            historial.append(self.alfa_valor)

            if mostrar and (it % 10 == 0 or it == self.n_iteraciones - 1):
                print(f"Iteración {it + 1:3d} | a = {a:.2f} "
                      f"| Mejor valor (alfa): {self.alfa_valor:.6f}")

        # Evaluar la última posición de la manada
        self.actualizar_lideres()
        historial[-1] = self.alfa_valor

        return self.alfa, self.alfa_valor, historial, posiciones


def graficar(funcion, limites, posiciones, historial):
    """Solo para problemas de 2 dimensiones."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mapa de contorno de la función
    a = np.linspace(limites[0], limites[1], 300)
    X, Y = np.meshgrid(a, a)
    Z = funcion(np.stack([X, Y], axis=-1))

    # Manada al inicio y al final
    for ax, m, titulo in zip(axes[:2], [0, len(posiciones) - 1],
                             ["Manada: inicio", "Manada: final"]):
        ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.8)
        ax.scatter(0, 0, marker="*", color="red", s=300, zorder=2,
                   label="Mínimo real (0, 0)")
        ax.scatter(posiciones[m][:, 0], posiciones[m][:, 1], color="white",
                   edgecolor="black", s=40, zorder=3, label="Lobos")
        ax.set_title(titulo)
        ax.set_xlim(limites)
        ax.set_ylim(limites)
        ax.legend(loc="upper right", fontsize=8)

    # Convergencia
    axes[2].plot(range(1, len(historial) + 1), historial, color="tab:green")
    axes[2].set_title(f"Convergencia (mejor = {historial[-1]:.6f})")
    axes[2].set_xlabel("Iteración")
    axes[2].set_ylabel("Mejor valor de la función (alfa)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultado_gwo.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    limites = (-5.12, 5.12)  # rango típico para Rastrigin

    gwo = LoboGris(
        funcion=rastrigin,
        dimensiones=2,
        limites=limites,
        n_lobos=30,
        n_iteraciones=100,
        semilla=1,
    )

    alfa, alfa_valor, historial, posiciones = gwo.ejecutar()

    print("\n=== RESULTADO ===")
    print("Mejor posición (lobo alfa):", np.round(alfa, 6))
    print(f"Valor de la función:        {alfa_valor:.6f}")
    print("(El mínimo real es f(0, 0) = 0)")

    graficar(rastrigin, limites, posiciones, historial)
