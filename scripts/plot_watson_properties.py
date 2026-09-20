"""
Plot two properties of the Watson distribution.

Figure 1: the angular mode a_p of the polar angle as a function of the concentration kappa, with an inset showing the
          polar angle probability density for four values of kappa.
Figure 2: the kappa-dependent expression whose root gives the maximum likelihood estimate of kappa.
"""
# Imports: third-party
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Imports: first-party
from jets import watson
from jets.paths import DIR_OUTPUT

plt.rcParams.update({
    "text.usetex":         True,
    "text.latex.preamble": r"\usepackage{gensymb}"})

# Initialize settings.
scalePD               = 100 # in 1; the inset shows probability densities in units of 10^-2 deg^-1
pathFigureAngularMode = DIR_OUTPUT / "plot_watson_angular_mode.pdf"
pathFigureMLEKappa    = DIR_OUTPUT / "plot_watson_mle_expression.pdf"


# Plot the angular mode as a function of the concentration.
kappas       = np.linspace(10, 0, num = 1000, endpoint = False)[::-1]
angularModes = np.degrees(np.arcsin(np.clip(1 / np.sqrt(2 * kappas), a_min = None, a_max = 1)))
angles       = np.linspace(0, 90, num = 900 + 1, endpoint = True)

plt.figure(figsize = (6, 3))
plt.plot(kappas, angularModes, c = "mediumseagreen")
plt.axvline(x = .5, ls = "--", c = "mediumseagreen")
plt.plot([2 / 3., 2 / 3.], [-1., 60.], c = "cornflowerblue", alpha = .5)
plt.plot([1., 1.], [-1., 45.], c = "goldenrod", alpha = .5)
plt.plot([2., 2.], [-1., 30.], c = "tomato", alpha = .5)
plt.plot([0., 2 / 3.], [60., 60.], c = "cornflowerblue", alpha = .5)
plt.plot([0., 1.], [45., 45.], c = "goldenrod", alpha = .5)
plt.plot([0., 2.], [30., 30.], c = "tomato", alpha = .5)
plt.xlim(0, 9)
plt.ylim(0, 90.5)
ax = plt.gca()
# Major ticks: 0, 1, 2, ..., 9
ax.set_xticks(np.arange(0, 10, 1))
# Minor ticks: every 1/6
ax.set_xticks(np.arange(0, 9 + 1e-9, 1 / 6), minor = True)
ax.tick_params(axis = "x", which = "minor")
ax.tick_params(axis = "x", which = "major")
plt.yticks([0, 15, 30, 45, 60, 75, 90], ["$0$", "$15$", "$30$", "$45$", "$60$", "$75$", "$90$"])
plt.text(0.6, 85, r"$\kappa = \frac{1}{2}$", ha = "left", va = "center")
plt.xlabel(r"Watson distribution concentration $\kappa\ (1)$")
plt.ylabel(r"angular mode $a_\mathrm{p} = \arcsin{\min{\left(\frac{1}{\sqrt{2\kappa}},1\right)}}\ (\degree)$")
# Put an inset in the top-right. The parameters 'width' and 'height' are fractions of the parent axes.
axInset = inset_axes(ax, width = "40%", height = "40%", loc = "upper right", borderpad = 1)
axInset.plot(angles, watson.PDsPolarAngle(angles, .5) * scalePD, c = "mediumseagreen", ls = "--")
axInset.plot(angles, watson.PDsPolarAngle(angles, 2 / 3.) * scalePD, c = "cornflowerblue", alpha = .5)
axInset.plot(angles, watson.PDsPolarAngle(angles, 1.) * scalePD, c = "goldenrod", alpha = .5)
axInset.plot(angles, watson.PDsPolarAngle(angles, 2.) * scalePD, c = "tomato", alpha = .5)
axInset.set_xlim(0, 90)
axInset.set_ylim(0, 2.)
axInset.set_xticks(np.linspace(0, 90, num = 6 + 1, endpoint = True))
axInset.set_xlabel(r"polar angle $a\ (\degree)$", fontsize = 8)
axInset.set_ylabel(r"PD $f_A(a)\ (10^{-2}\,\mathrm{deg}^{-1})$", fontsize = 8)
plt.subplots_adjust(left = .1, bottom = 0.14, right = .98, top = .98)
plt.savefig(pathFigureAngularMode)
plt.close()
print(f"Saved plot to '{pathFigureAngularMode}'.")


# Plot the expression whose root is the maximum likelihood estimate of the concentration.
kappas              = np.linspace(-9, 9, num = 1800 + 1, endpoint = True)
MLEExpressionsKappa = watson.MLEExpressionKappa(kappas)

plt.figure(figsize = (6, 3))
plt.plot(kappas, MLEExpressionsKappa, c = "mediumseagreen")
plt.xlim(-9, 9)
plt.xticks(np.linspace(-9, 9, num = 7, endpoint = True), ["$-9$", "$-6$", "$-3$", "$0$", "$3$", "$6$", "$9$"])
plt.ylim(-0.02, 1.02)
plt.text(8.8, 1 - .05, r"$\kappa \to \infty{:}\ 1$", ha = "right", va = "center")
plt.text(8.8, .05, r"$\kappa \to -\infty{:}\ 0$", ha = "right", va = "center")
plt.text(8.8, 1 / 3. + .05, r"$\kappa = 0{:}\ \frac{1}{3}$", ha = "right", va = "center")
plt.axhline(0, c = "mediumseagreen", linestyle = ":")
plt.axhline(1 / 3., c = "mediumseagreen", linestyle = ":")
plt.axhline(1, c = "mediumseagreen", linestyle = ":")
plt.xlabel(r"Watson distribution concentration $\kappa\ (1)$")
plt.ylabel(r"$\frac{\exp{\kappa}}{\sqrt{\pi \kappa}\mathrm{erfi}\sqrt{\kappa}}-\frac{1}{2\kappa}\ (1)$")
plt.subplots_adjust(left = .1, bottom = 0.14, right = .98, top = .98)
plt.savefig(pathFigureMLEKappa)
plt.close()
print(f"Saved plot to '{pathFigureMLEKappa}'.")
