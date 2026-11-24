"""
Optimización por Enjambre de Partículas (Particle Swarm Optimization, PSO)
Aplicado a la minimización de funciones de prueba (Rastrigin)

Materia: Algoritmos Bioinspirados
Alumno: Dante Yael Macías Rentería

Idea bioinspirada:
  Se inspira en el comportamiento de parvadas de aves y bancos de peces.
  Cada individuo (partícula) se mueve por el espacio de búsqueda guiado por:
    - su propia experiencia: el mejor lugar que ELLA ha encontrado (pbest)
    - la experiencia del grupo: el mejor lugar que encontró todo el enjambre (gbest)
  Así, sin un líder central, el enjambre converge hacia las mejores zonas.

En el algoritmo, cada partícula i tiene una posición x_i y una velocidad v_i.
En cada iteración se actualizan así:

    v_i = w * v_i                        -> inercia (sigue moviéndose como iba)
        + c1 * r1 * (pbest_i - x_i)      -> componente cognitiva (memoria propia)
        + c2 * r2 * (gbest  - x_i)       -> componente social (aprende del grupo)

    x_i = x_i + v_i

  w      : peso de inercia
  c1, c2 : coeficientes de aceleración (cognitivo y social)
  r1, r2 : números aleatorios uniformes en [0, 1]
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


class EnjambreParticulas:
    def __init__(self, funcion, dimensiones, limites, n_particulas=30,
                 n_iteraciones=100, w=0.7, c1=1.5, c2=1.5, semilla=None):
        """
        funcion       : función objetivo a minimizar
        dimensiones   : número de variables del problema
        limites       : (mínimo, máximo) del espacio de búsqueda
        n_particulas  : tamaño del enjambre
        n_iteraciones : número de iteraciones
        w             : peso de inercia
        c1            : coeficiente cognitivo (atracción a su mejor posición)
        c2            : coeficiente social (atracción a la mejor del enjambre)
        """
        self.f = funcion
        self.d = dimensiones
        self.lim_inf, self.lim_sup = limites
        self.n_particulas = n_particulas
        self.n_iteraciones = n_iteraciones
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(semilla)

        # Velocidad máxima: 20% del rango, para que no "salgan disparadas"
        self.v_max = 0.2 * (self.lim_sup - self.lim_inf)

        # 1) Posiciones iniciales aleatorias dentro de los límites
        self.x = self.rng.uniform(self.lim_inf, self.lim_sup,
                                  size=(n_particulas, dimensiones))
        # 2) Velocidades iniciales pequeñas
        self.v = self.rng.uniform(-self.v_max, self.v_max,
                                  size=(n_particulas, dimensiones))

        # 3) Mejor posición personal de cada partícula (al inicio, donde está)
        self.pbest = self.x.copy()
        self.pbest_valor = self.f(self.x)

        # 4) Mejor posición global del enjambre
        idx = np.argmin(self.pbest_valor)
        self.gbest = self.pbest[idx].copy()
        self.gbest_valor = self.pbest_valor[idx]

    def actualizar_velocidad(self):
        r1 = self.rng.random((self.n_particulas, self.d))
        r2 = self.rng.random((self.n_particulas, self.d))

        inercia = self.w * self.v
        cognitiva = self.c1 * r1 * (self.pbest - self.x)
        social = self.c2 * r2 * (self.gbest - self.x)

        self.v = inercia + cognitiva + social
        # Limitar la velocidad
        self.v = np.clip(self.v, -self.v_max, self.v_max)

    def actualizar_posicion(self):
        self.x = self.x + self.v
        # Mantener las partículas dentro del espacio de búsqueda
        self.x = np.clip(self.x, self.lim_inf, self.lim_sup)

    def actualizar_mejores(self):
        valores = self.f(self.x)

        # Actualizar pbest de las partículas que mejoraron
        mejoraron = valores < self.pbest_valor
        self.pbest[mejoraron] = self.x[mejoraron]
        self.pbest_valor[mejoraron] = valores[mejoraron]

        # Actualizar gbest si alguna superó al mejor global
        idx = np.argmin(self.pbest_valor)
        if self.pbest_valor[idx] < self.gbest_valor:
            self.gbest = self.pbest[idx].copy()
            self.gbest_valor = self.pbest_valor[idx]

    def ejecutar(self, mostrar=True):
        historial = []
        posiciones = [self.x.copy()]  # para graficar cómo se mueve el enjambre

        for it in range(self.n_iteraciones):
            self.actualizar_velocidad()
            self.actualizar_posicion()
            self.actualizar_mejores()

            historial.append(self.gbest_valor)
            posiciones.append(self.x.copy())

            if mostrar and (it % 10 == 0 or it == self.n_iteraciones - 1):
                print(f"Iteración {it + 1:3d} | Mejor valor: {self.gbest_valor:.6f}")

        return self.gbest, self.gbest_valor, historial, posiciones


def graficar(funcion, limites, posiciones, historial):
    """Solo para problemas de 2 dimensiones."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mapa de contorno de la función
    a = np.linspace(limites[0], limites[1], 300)
    X, Y = np.meshgrid(a, a)
    Z = funcion(np.stack([X, Y], axis=-1))

    # Enjambre al inicio y al final
    for ax, m, titulo in zip(axes[:2], [0, len(posiciones) - 1],
                             ["Inicio", "Final"]):
        ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.8)
        ax.scatter(0, 0, marker="*", color="red", s=300, zorder=2,
                   label="Mínimo real (0, 0)")
        ax.scatter(posiciones[m][:, 0], posiciones[m][:, 1], color="white",
                   edgecolor="black", s=40, zorder=3, label="Partículas")
        ax.set_title(f"Enjambre: {titulo}")
        ax.set_xlim(limites)
        ax.set_ylim(limites)
        ax.legend(loc="upper right", fontsize=8)

    # Convergencia
    axes[2].plot(range(1, len(historial) + 1), historial, color="tab:green")
    axes[2].set_title(f"Convergencia (mejor = {historial[-1]:.6f})")
    axes[2].set_xlabel("Iteración")
    axes[2].set_ylabel("Mejor valor de la función")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("resultado_pso.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    limites = (-5.12, 5.12)  # rango típico para Rastrigin

    pso = EnjambreParticulas(
        funcion=rastrigin,
        dimensiones=2,
        limites=limites,
        n_particulas=30,
        n_iteraciones=150,
        w=0.7,
        c1=1.5,
        c2=1.5,
        semilla=1,
    )

    mejor, mejor_valor, historial, posiciones = pso.ejecutar()

    print("\n=== RESULTADO ===")
    print("Mejor posición encontrada:", np.round(mejor, 6))
    print(f"Valor de la función:       {mejor_valor:.6f}")
    print("(El mínimo real es f(0, 0) = 0)")

    graficar(rastrigin, limites, posiciones, historial)
