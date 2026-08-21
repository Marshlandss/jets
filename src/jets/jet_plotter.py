# === imports ===

import math
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from astropy.io import fits
from astropy.visualization import ImageNormalize, ManualInterval, SqrtStretch
from astropy.convolution import Gaussian1DKernel, convolve

from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks

# === JetPlotter class ===

class JetPlotter:
    def __init__(self, jet_system, save_a, save_sa, save_m, save_format, axis_font_size):
        self.jet = jet_system

        self.save_a = save_a
        self.save_sa = save_sa
        self.save_m = save_m
        self.save_format = save_format
        self.axis_font_size = axis_font_size
                
        self.save_plots()

    def save_plots(self):
        """
        Saves plots based on self.folder_status.
        """

        save_location_mapping = {
            "a"         : self.save_a,
            "sa"        : self.save_sa,
            "m"         : self.save_m,
        }

        for status, save_location in save_location_mapping.items():
            if self.jet.adjustment_status == status :

                num_of_rows = len(self.jet.light_values_2d)
                fig = plt.figure(figsize=(11, 4 * num_of_rows))

                gs = fig.add_gridspec(len(self.jet.light_values_2d), 3, width_ratios = [1, .1, 1.2], hspace = .01)

                for i in range(len(self.jet.light_values_2d)):

                    scatter_axis = fig.add_subplot(gs[i, 2])
                    overlay_axis = fig.add_subplot(gs[i, 0])

                    self.create_scatter_ax(scatter_axis, self.jet.light_values_2d[i], self.jet.light_values_convolved_2d[i], self.jet.threshold_list[i], self.jet.best_angle_index_list[i])
                    self.create_overlay_ax(overlay_axis, self.jet.best_angle_index_list[i], self.jet.radius_list[i])

                    scatter_axis.get_xaxis().set_visible(False)
                    overlay_axis.get_xaxis().set_visible(False)

                    if i == 0:

                        v_min       = np.nanpercentile(self.jet.image, 10)
                        v_max       = np.nanpercentile(self.jet.image, 99.9)
                        object_norm = ImageNormalize(self.jet.image, interval=ManualInterval(vmin=v_min, vmax=v_max), stretch=SqrtStretch())

                        # Draw subplot components
                        h, w = self.jet.image.shape
                        crop_img = self.jet.image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
                        flipped_img = np.flipud(crop_img)

                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                        inset_ax = inset_axes(overlay_axis, width = "30%", height = "30%", loc = "upper right", borderpad = 0.2 )

                        center_x, center_y = (flipped_img.shape[1]-1) / 2, (flipped_img.shape[0]-1) / 2

                        inset_ax.imshow(flipped_img, norm=object_norm, cmap = "inferno")

                        for side in ['top', 'bottom', 'left', 'right']:
                            inset_ax.spines[side].set_visible(True)
                            inset_ax.spines[side].set_edgecolor("grey")
                            inset_ax.spines[side].set_linewidth(1)
                            inset_ax.spines[side].set_zorder(10)  # Bring border above image

                        inset_ax.set_xticks([])
                        inset_ax.set_yticks([])

                        circle = patches.Circle((center_x, center_y), self.jet.image.shape[0]*.01, facecolor="red", linewidth = 1, edgecolor="orange", fill=True, )
                        inset_ax.add_patch(circle)

                    if i == len(self.jet.light_values_2d) - 1:

                        if self.jet.adjustment_status == "m":
                            overlay_axis.plot(self.jet.manual_x, self.jet.manual_y, color = "lightgreen", linewidth=2)
                            scatter_axis.axvline(self.jet.best_angle, color = "green", linewidth = 1, linestyle = "--")
                            self.add_text_to_ax(overlay_axis, "Manual")
                        elif self.jet.adjustment_status == "sa":
                            num = int(self.jet.radius_list[i]) // self.jet.increase_radius_step
                            if num == 1 and self.jet.radius_list[0] == self.jet.radius_list[1]:
                                self.add_text_to_ax(overlay_axis, "Selected: 0")
                            else:
                                self.add_text_to_ax(overlay_axis, "Selected: " + str(num))

                        else:
                            self.add_text_to_ax(overlay_axis, int(self.jet.radius_list[i]) // self.jet.increase_radius_step)

                        scatter_axis.get_xaxis().set_visible(True)
                        overlay_axis.get_xaxis().set_visible(True)

                    else:
                        self.add_text_to_ax(overlay_axis, i)

                fig.savefig(save_location, format = self.save_format, bbox_inches = "tight")
                # plt.show()
                plt.close()

    def create_scatter_ax(self, scatter_ax, light_values, light_values_convolved, threshold, best_angle_index):
        """
        Creates a scatter plot of the data and saves it to self.scatter_plot_data
        """

        # === Formating ===
        scatter_ax.set_ylabel(r'mean SI $I_\nu$' + '\n' r'$(\mathrm{mJy\ arcmin^{-2}})$', fontsize = self.axis_font_size, linespacing = 1.2)
        scatter_ax.set_xlabel(r"line segment PA ($\degree$)", fontsize = self.axis_font_size)

        # === Plot data ===
        # --- Plot light values data ---
        angles_degrees      = self.jet.angles * 180 / np.pi
        scatter_ax.scatter(angles_degrees, light_values, color="gainsboro")

        # --- Plot light values convolved data ---
        x                   = list(range(0, 180))
        y                   = light_values_convolved
        x_smooth            = np.linspace(min(x), max(x), 300)
        spl                 = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
        y_smooth            = spl(x_smooth)
        scatter_ax.plot(x_smooth, y_smooth, color="dimgrey", linewidth=2)

        # --- Vertical (best angle) and horizonal (threshold) lines ---
        scatter_ax.axhline(y = threshold, color = "purple", linewidth = 1, linestyle = "--")
        best_angle = np.degrees(self.jet.angles[best_angle_index])
        scatter_ax.axvline(x = best_angle, color = "cadetblue", linewidth = 1, linestyle = "--")
        scatter_ax.axvline(x = best_angle, color = "cadetblue", linewidth = 1, linestyle = "--")

        # --- Mark absolute and local maxima ---
        maxima_indices = self.jet.find_peaks(light_values_convolved)

        scatter_ax.scatter(angles_degrees[maxima_indices], light_values_convolved[maxima_indices],
                        color="purple", zorder=10, s=60)
        scatter_ax.scatter(angles_degrees[best_angle_index], light_values_convolved[best_angle_index],
                        color="cyan", zorder=10, s=50)

        # --- Ticks and Labels ---
        fig = scatter_ax.get_figure()
        fig.canvas.draw()

        yticks_max = max(scatter_ax.get_yticks())
        yticks_min = min(scatter_ax.get_yticks())
        spacer = (yticks_max - yticks_min) * 0.15

        yticks = np.linspace(yticks_min + spacer, yticks_max - spacer, 5)
        scatter_ax.set_yticks(yticks)
        scatter_ax.set_yticklabels([f"{i * 1000:.0f}" for i in yticks])

        xticks = np.linspace(0, 180, 19)
        scatter_ax.set_xticks(xticks)
        scatter_ax.set_xlim(0, 180)
        xlab = []
        for i in range(0, len(xticks)):
            if i % 3 == 0:
                xlab.append(f"{(i*10):.0f}")
            else:
                xlab.append("")

        scatter_ax.set_xticklabels(xlab)

    def create_overlay_ax(self, overlay_ax, best_angle_index, radius):
        """
        Creates a plot of the cutout image and overlays the line of the best angle. Saves to self.overlay_image_data.
        """

        # Formating
        dpi = 1000

        # Calculate v_min and v_max

        v_min       = np.nanpercentile(self.jet.image, 10)
        v_max       = np.nanpercentile(self.jet.image, 99.9)
        object_norm = ImageNormalize(self.jet.image, interval=ManualInterval(vmin=v_min, vmax=v_max), stretch=SqrtStretch())

        # overlay_figure, overlay_ax = plt.subplots(figsize=(7.3, 6))

        overlay_ax.set_xlim(0, self.jet.image.shape[1])
        overlay_ax.set_ylim(0, self.jet.image.shape[0])
        overlay_ax.set_xlabel(r"right ascension ($\degree$)", fontsize = self.axis_font_size)
        overlay_ax.set_ylabel(r"declination ($\degree$)", fontsize = self.axis_font_size, linespacing = 0.95)

        overlay_ax.patch.set_facecolor('purple')

        # overlay_figure.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.12)

        # Converting ax units
        pixel_size_arcmin = self.calculate_pixel_size_deg(self.jet.image, self.jet.image_width_arcmins) # in deg

        shift = self.format_ticks(self.jet.image, pixel_size_arcmin, self.jet.declination, overlay_ax, 1, 0, self.jet.declination)
        self.format_ticks(self.jet.image, pixel_size_arcmin, self.jet.right_ascension, overlay_ax, 0, shift, self.jet.declination)

        # Draw main plot components
        overlay_ax.imshow(self.jet.image, origin="lower", cmap="inferno", norm=object_norm)

        x, y            = self.jet._create_coordinate_list(self.jet.angles[best_angle_index], radius)
        overlay_ax.plot(x, y, color="cyan", linewidth=2)

        circle          = plt.Circle(xy=(self.jet.central_point[0], self.jet.central_point[1]), radius = radius, fill = True,
                                facecolor = "white",
                                edgecolor = "green", alpha=0.3)
        central_point   = plt.Circle(xy=(self.jet.central_point[0], self.jet.central_point[1]), radius=self.jet.image.shape[0]*.01, fill=True,
                                        facecolor="red", zorder=10)
        overlay_ax.add_artist(circle)
        overlay_ax.add_artist(central_point)

    def calculate_pixel_size_deg(self, img, angular_length):
        pixel_size = angular_length / img.shape[0]  # in arcmin
        pixel_size = pixel_size / 60  # in deg

        return pixel_size

    def format_ticks(self, image, pixel_size_arcmin, center_angle, ax, id, shift_value_dec, dec):
        ticks = np.linspace(0, image.shape[0], 5, endpoint = True)
        ticks_converted = ticks * pixel_size_arcmin

        if shift_value_dec != 0:
            shift = shift_value_dec / np.cos(np.radians(dec))
        else:
            shift = ticks_converted[1]
        ticks = (ticks[1:4])
        tick_labels = []

        if id == 0:
            ticks_converted = [f"{float(center_angle + shift):.2f}", f"{float(center_angle):.2f}", f"{float(center_angle - shift):.2f}"]
            ax.set_xticks(ticks)
            tick_labels = ax.set_xticklabels(ticks_converted)

        else:
            ticks_converted = [f"{float(center_angle - shift):.2f}", f"{float(center_angle):.2f}", f"{float(center_angle + shift):.2f}"]
            ax.set_yticks(ticks)
            tick_labels = ax.set_yticklabels(ticks_converted)


        tick_labels[1].set_bbox(dict(facecolor='none', boxstyle='round,pad=0.3', linewidth = .5, edgecolor = "white"))


        return shift

    def add_text_to_ax(self, ax, text):
            ax.text(
                self.jet.image.shape[0] * 0.05, self.jet.image.shape[1] * 0.88,
                text,
                fontsize = self.axis_font_size,
                color = "white",
                bbox = dict(facecolor = "gray", edgecolor = "none", boxstyle = "round,pad=0.3", alpha = 0.5)
            )