"""
Algoritmo Genético (Genetic Algorithm, GA)
Aplicado a la minimización de funciones de prueba (Rastrigin)

Materia: Algoritmos Bioinspirados
Alumno: Dante Yael Macías Rentería

Idea bioinspirada:
  Se basa en la teoría de la evolución de Darwin y en la genética.
  Una población de individuos compite: los más aptos tienen más probabilidad
  de reproducirse y pasar sus genes a la siguiente generación. La cruza mezcla
  genes de dos padres y la mutación introduce cambios aleatorios, lo que
  mantiene la diversidad. Con las generaciones, la población "evoluciona"
  hacia mejores soluciones.

En el algoritmo:
  - Cromosoma: cadena de bits (0 y 1) que codifica las variables del problema.
  - Aptitud (fitness): qué tan buena es la solución. Como queremos MINIMIZAR,
    un valor de f(x) más pequeño significa mejor aptitud.
  - Operadores genéticos en cada generación:
        1) Selección por torneo: se eligen k individuos al azar y gana el mejor.
        2) Cruza de un punto: se corta a los dos padres en un punto aleatorio
           y se intercambian las partes.
        3) Mutación: cada bit se invierte (0 <-> 1) con una probabilidad pequeña.
        4) Elitismo: los mejores individuos pasan directo a la siguiente
           generación para no perder la mejor solución encontrada.
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


class AlgoritmoGenetico:
    def __init__(self, funcion, dimensiones, limites, bits_por_variable=16,
                 tam_poblacion=50, n_generaciones=100, prob_cruza=0.9,
                 prob_mutacion=None, tam_torneo=3, n_elite=2, semilla=None):
        """
        funcion           : función objetivo a minimizar
        dimensiones       : número de variables del problema
        limites           : (mínimo, máximo) del espacio de búsqueda
        bits_por_variable : precisión de la codificación binaria
        tam_poblacion     : número de individuos
        n_generaciones    : número de generaciones
        prob_cruza        : probabilidad de que dos padres se crucen
        prob_mutacion     : probabilidad de invertir cada bit
                            (por defecto 1 / longitud del cromosoma)
        tam_torneo        : individuos que compiten en cada torneo
        n_elite           : mejores individuos que pasan directo
        """
        self.f = funcion
        self.d = dimensiones
        self.lim_inf, self.lim_sup = limites
        self.bits = bits_por_variable
        self.long_cromosoma = dimensiones * bits_por_variable
        self.tam_poblacion = tam_poblacion
        self.n_generaciones = n_generaciones
        self.prob_cruza = prob_cruza
        self.prob_mutacion = (prob_mutacion if prob_mutacion is not None
                              else 1.0 / self.long_cromosoma)
        self.tam_torneo = tam_torneo
        self.n_elite = n_elite
        self.rng = np.random.default_rng(semilla)

        # Pesos para pasar de binario a entero: [2^(b-1), ..., 2, 1]
        self.potencias = 2 ** np.arange(self.bits - 1, -1, -1)

        # Población inicial: cromosomas de bits aleatorios
        self.poblacion = self.rng.integers(0, 2, size=(tam_poblacion,
                                                       self.long_cromosoma))

    def decodificar(self, poblacion):
        """Convierte cada cromosoma binario en un vector de números reales.
           entero = suma(bit * 2^posición)
           real   = lim_inf + entero * (lim_sup - lim_inf) / (2^bits - 1)
        """
        genes = poblacion.reshape(len(poblacion), self.d, self.bits)
        enteros = genes @ self.potencias
        escala = (self.lim_sup - self.lim_inf) / (2 ** self.bits - 1)
        return self.lim_inf + enteros * escala

    def evaluar(self, poblacion):
        """Valor de la función objetivo de cada individuo (menor = mejor)."""
        return self.f(self.decodificar(poblacion))

    def seleccion_torneo(self, valores):
        """Elige k individuos al azar y regresa el índice del mejor."""
        competidores = self.rng.choice(self.tam_poblacion, self.tam_torneo,
                                       replace=False)
        return competidores[np.argmin(valores[competidores])]

    def cruza(self, padre1, padre2):
        """Cruza de un punto."""
        if self.rng.random() < self.prob_cruza:
            punto = self.rng.integers(1, self.long_cromosoma)
            hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
            hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
            return hijo1, hijo2
        return padre1.copy(), padre2.copy()

    def mutacion(self, individuo):
        """Invierte cada bit con probabilidad prob_mutacion."""
        mascara = self.rng.random(self.long_cromosoma) < self.prob_mutacion
        individuo[mascara] = 1 - individuo[mascara]
        return individuo

    def nueva_generacion(self, valores):
        # Elitismo: los mejores pasan sin cambios
        orden = np.argsort(valores)
        nueva = [self.poblacion[i].copy() for i in orden[:self.n_elite]]

        # El resto se genera con selección, cruza y mutación
        while len(nueva) < self.tam_poblacion:
            padre1 = self.poblacion[self.seleccion_torneo(valores)]
            padre2 = self.poblacion[self.seleccion_torneo(valores)]
            hijo1, hijo2 = self.cruza(padre1, padre2)
            nueva.append(self.mutacion(hijo1))
            if len(nueva) < self.tam_poblacion:
                nueva.append(self.mutacion(hijo2))

        self.poblacion = np.array(nueva)

    def ejecutar(self, mostrar=True):
        hist_mejor = []
        hist_promedio = []
        poblaciones = [self.decodificar(self.poblacion)]  # para graficar

        mejor_cromosoma = None
        mejor_valor = float("inf")

        for gen in range(self.n_generaciones):
            valores = self.evaluar(self.poblacion)

            idx = int(np.argmin(valores))
            if valores[idx] < mejor_valor:
                mejor_valor = valores[idx]
                mejor_cromosoma = self.poblacion[idx].copy()

            hist_mejor.append(mejor_valor)
            hist_promedio.append(valores.mean())

            if mostrar and (gen % 10 == 0 or gen == self.n_generaciones - 1):
                print(f"Generación {gen + 1:3d} | Mejor: {mejor_valor:.6f} "
                      f"| Promedio: {valores.mean():.4f}")

            if gen < self.n_generaciones - 1:
                self.nueva_generacion(valores)
                poblaciones.append(self.decodificar(self.poblacion))

        mejor_x = self.decodificar(mejor_cromosoma[None, :])[0]
        return mejor_cromosoma, mejor_x, mejor_valor, hist_mejor, hist_promedio, poblaciones


def graficar(funcion, limites, poblaciones, hist_mejor, hist_promedio):
    """Solo para problemas de 2 dimensiones."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mapa de contorno de la función
    a = np.linspace(limites[0], limites[1], 300)
    X, Y = np.meshgrid(a, a)
    Z = funcion(np.stack([X, Y], axis=-1))

    # Población en la primera y en la última generación
    for ax, m, titulo in zip(axes[:2], [0, len(poblaciones) - 1],
                             ["Generación inicial", "Generación final"]):
        ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.8)
        ax.scatter(0, 0, marker="*", color="red", s=300, zorder=2,
                   label="Mínimo real (0, 0)")
        ax.scatter(poblaciones[m][:, 0], poblaciones[m][:, 1], color="white",
                   edgecolor="black", s=40, zorder=3, label="Individuos")
        ax.set_title(titulo)
        ax.set_xlim(limites)
        ax.set_ylim(limites)
        ax.legend(loc="upper right", fontsize=8)

    # Convergencia: mejor individuo y promedio de la población
    gens = range(1, len(hist_mejor) + 1)
    axes[2].plot(gens, hist_mejor, color="tab:green", label="Mejor")
    axes[2].plot(gens, hist_promedio, color="tab:orange", label="Promedio")
    axes[2].set_title(f"Convergencia (mejor = {hist_mejor[-1]:.6f})")
    axes[2].set_xlabel("Generación")
    axes[2].set_ylabel("Valor de la función")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("resultado_ga.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    limites = (-5.12, 5.12)  # rango típico para Rastrigin

    ga = AlgoritmoGenetico(
        funcion=rastrigin,
        dimensiones=2,
        limites=limites,
        bits_por_variable=16,
        tam_poblacion=50,
        n_generaciones=100,
        prob_cruza=0.9,
        prob_mutacion=None,   # None = 1 / longitud del cromosoma
        tam_torneo=3,
        n_elite=2,
        semilla=1,
    )

    cromosoma, mejor_x, mejor_valor, hist_mejor, hist_promedio, poblaciones = ga.ejecutar()

    print("\n=== RESULTADO ===")
    print("Mejor cromosoma:", "".join(map(str, cromosoma)))
    print("Mejor posición encontrada:", np.round(mejor_x, 6))
    print(f"Valor de la función:       {mejor_valor:.6f}")
    print("(El mínimo real es f(0, 0) = 0)")

    graficar(rastrigin, limites, poblaciones, hist_mejor, hist_promedio)
