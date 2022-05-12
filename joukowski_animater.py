import spingen
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import spin_animater
import animater
import joukowski
import __main__

class JoukowskiAnimation(spin_animater.SFlowMation):


    def __init__(self,  nu = -0.22 +0.125j, gamma=10):
        super().__init__(gamma)
        self.nu = nu
        self.transformer = joukowski.JoukowskiFlow(gamma, nu)
        self.a = np.abs(self.nu-1)

    def generate_trajectories(self, nb_frames, nb_particles):
        trajs = super().generate_trajectories(nb_frames, nb_particles)
        return self.transformer.coodmap(trajs + self.nu)

    def populate_equilines(self):
        super().populate_equilines()
        for i in range(len(self.streams)):
            self.streams[i] = (self.streams[i][0], self.transformer.coodmap(self.streams[i][1] + self.nu) )
        for i in range(len(self.equiphi)):
            self.equiphi[i] = (self.equiphi[i][0], self.transformer.coodmap(self.equiphi[i][1] + self.nu) )

    def _obstacle_eqn(self, t):
        radius = np.abs(self.nu - 1)
        circle = self.a*(np.cos(np.real(t)) + 1j*np.sin(np.real(t))) + self.nu
        return self.transformer.coodmap(circle)

JkAnime = JoukowskiAnimation()
JkAnime.show_movie(0.0001, 200)