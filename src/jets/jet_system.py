
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

# === JetSystem class ===

class JetSystem:

    def __init__(self, path, angles,
                 right_ascensions, declinations, length_angular_means, redshifts, are_Spectroscopic_bool_list,
                 percentage, nan_percentage_cutoff,
                 excel_file, save_auto_loc, save_manual_loc, save_some_manual_loc,
                 saving_plots, pixel_shift, save_format,
                 manual_list, some_manual_list, axis_font_size):

        # === Attributes ===
        # --- External ---
        self.right_ascensions               = right_ascensions
        self.declinations                   = declinations
        self.length_angular_means           = length_angular_means
        self.redshifts                      = redshifts
        self.angles                         = angles
        self.saving_plots                   = saving_plots
        self.some_manual_list               = some_manual_list
        self.manual_list                    = manual_list
        self.axis_font_size                 = axis_font_size
        self.are_Spectroscopic_bool_list    = are_Spectroscopic_bool_list
        self.save_format                    = save_format

        # --- Initializing ---
        self.path                       = path
        self.right_ascension            = 0
        self.declination                = 0
        self.light_values_2d            = list()
        self.light_values_convolved_2d  = list()
        self.max_light_sum              = 0
        self.best_angle_index_list      = []
        self.best_angle                 = 0
        self.radius_list                = []
        self.percentage                 = percentage
        self.nan_percentage_cutoff      = nan_percentage_cutoff
        self.threshold_list             = []
        self.jet_width_pixels           = 0
        self.uncertainty                = 0
        self.nan_percentage             = 1
        self.adjustment_status          = "auto"
        self.manual_x                   = []
        self.manual_y                   = []
        self.increase_radius_step       = 0

        file_name                       = self.path.split('/')[-1]
        fits_data                       = file_name.split("_")
        self.right_ascension            = float(fits_data[0])
        self.declination                = float(fits_data[1])
        self.image_width_arcmins        = float(fits_data[2][0: fits_data[2].index("A")])

        hdu_list                        = fits.open(self.path)
        self.image                      = hdu_list[0].data
        self.image                      = self.image.copy()/(self.beamGaussian(6,6)*3600)

        self.image                      = np.roll(np.roll(self.image, -pixel_shift, axis=1), -pixel_shift, axis=0)
        hdu_list.close()

        self.image_side_length          = self.image.shape[0]
        self.central_point              = [(self.image_side_length - 1) / 2., (self.image_side_length - 1) / 2.]
        self.nan_percentage             = (np.sum(np.isnan(self.image)) / (self.image.shape[0] * self.image.shape[1])) * 100

        self.pixel_width_arcmins        = self.image_width_arcmins / self.image_side_length

        file_name                       = file_name[:-5] + "_jet_orientation." + save_format

        self.save_auto_loc              = save_auto_loc + file_name
        self.save_manual_loc            = save_manual_loc + file_name
        self.save_some_manual_loc       = save_some_manual_loc + file_name
        self.excel_file                 = excel_file

        # === Function Calls ===

        if not self._check_if_file_exists_in_correct_location():
            print("Creating new files...")

            index                           = self._find_catalogue_match_index(self.right_ascension, self.declination)
            length_angular_mean             = self.length_angular_means[index]
            redshift                        = self.redshifts[index]
            are_Spectroscopic_bool          = self.are_Spectroscopic_bool_list[index]
            resolution_arcsec               = 6

            if length_angular_mean == 0 or length_angular_mean == None:
                print("SKIPPED - No match for jet system:", self.right_ascension, "+", self.declination)

            else :

                self.jet_width_pixels       = length_angular_mean / self.pixel_width_arcmins # in pixels
                self.increase_radius_step   = int((self.jet_width_pixels / 2) /7)
                if (length_angular_mean * 60)/2 * 0.1 < 6:
                    print("Starting radius too small; increasing to FWHM.")
                    self.starting_radius = int((resolution_arcsec / 60) / self.pixel_width_arcmins) # in pixels
                else:
                    print("Starting radius good.")
                    self.starting_radius        = int(self.jet_width_pixels / 20) # in pixels
                print("Starting radius:", self.starting_radius)

                self.radius_list.append(self.starting_radius)
                self._find_brightest()

                if self.nan_percentage > self.nan_percentage_cutoff:
                    print("WARNING - ", self.nan_percentage, "% of light values are NaN:", self.right_ascension, "+", self.declination)

                self._find_best_radius()
                print("Jet angle: ", self.best_angle)
                print("status: ", self.adjustment_status)
                print("redshift: ", redshift)
                # self._record_jet_data(redshift, are_Spectroscopic_bool) commented for testing only

                if self.saving_plots:
                    print("Saving...")
                    self._save_plots()

    # === Functions ===
    # --- Directly called in constructor ---

    def beamGaussian(self, FWHMMajor, FWHMMinor, unitsArcsec = True):
        """
        Calculate the solid angle of a restoring Gaussian PSF, with FWHMs 'FWHMMajor' and 'FWHMMinor'.
        If 'unitsArcsec' is True, the FWHMs are assumed to be in arcseconds, otherwise the FWHMs are assumed to be in degrees.
        Returns the beam in square degrees.
        For a derivation, see Section 3.3.3 of https://www.cv.nrao.edu/~sransom/web/Ch3.html.
        """
        if (unitsArcsec):
            FWHMMajor /= 3600 # in deg
            FWHMMinor /= 3600 # in deg

        return np.pi * FWHMMajor * FWHMMinor / (4 * np.log(2)) # in sq deg

    def _check_if_file_exists_in_correct_location(self):
        """
        Identifies if the current jet system has already been saved in to one of the folders.
        Checks if its adjustment_status aligns with the folder it has been saved to.
        """

        self._check_adjustment_status()

        mapping = {
            "auto"                 : self.save_auto_loc,
            "manual"               : self.save_manual_loc,
            "some_manual"          : self.save_some_manual_loc,

        }

        for status, save_location in mapping.items():
            if os.path.exists(save_location) :
                print("File exists in location:", save_location)
                if self.adjustment_status == status :
                    # File is saved to the correct folder
                    print("File exists in correct location.")
                    return True
                else:
                    # File exists in a folder, however not the correct folder
                    print("File does not exist in correct location... Removing files...")
                    os.remove(save_location)
                    return False

        # File does not exist in any folders
        print("File does not exist yet.")
        return False

    def _find_brightest(self):
        """
        Finds the angle with the brightest light value sum.
        Sets: self.light_values, self.light_values_convolved, self.max_light_sum, self.best_angle_index,
        self.best_angle, self.threshold, self.threshold2
        """
        light_values = []

        for angle in self.angles:
            x_values, y_values          = self._create_coordinate_list(angle, self.radius_list[-1])
            x_values                    = [int(np.round(value)) for value in x_values]
            y_values                    = [int(np.round(value)) for value in y_values]

            light_values.append(np.mean(self.image[y_values, x_values]))

        self.light_values_2d.append(np.nan_to_num(light_values))

        # Smoothes data
        gauss                           = Gaussian1DKernel(stddev=5)

        self.light_values_convolved_2d.append(convolve(self.light_values_2d[-1], gauss, boundary="wrap"))
        self.best_angle_index_list.append(np.argmax(self.light_values_convolved_2d[-1]))

        self.best_angle                 = int(self.angles[self.best_angle_index_list[-1]] * 180 / np.pi)
        self.max_light_sum              = np.amax(self.light_values_convolved_2d[-1])

        self.threshold_list.append(self.max_light_sum * self.percentage)

    def _find_best_radius(self):
        """

        """

        # Distance from self.best_angle to the furthest peak above self.threshold
        self.uncertainty = self._find_uncertainty(self.threshold_list[-1])

        for i in range(self.increase_radius_step , int(self.jet_width_pixels / 2) + 1, self.increase_radius_step):
            if self.uncertainty != 0:
                if i < self.starting_radius:
                    self._adjust_radius(self.starting_radius)
                else:
                    self._adjust_radius(i)
                self.uncertainty    = self._find_uncertainty(self.threshold_list[-1])
            else:
                break

        # Applies changes based on manually inputted lists
        self._make_adjustments()

    def get_subtraction_used(self):

        if self.path.endswith("_sub.fits"):
            return True
        else:
            return False

    def _record_jet_data(self, redshift, are_Spectroscopic_bool):
        """
        Records/updates an entry to the Excel sheet.
        """

        # Existing Dataframe
        df_existing     = pd.read_excel(self.excel_file)

        # Checks if an entry for this jet system already exists and deletes it.
        match           = (df_existing["right_ascension (deg)"] == self.right_ascension) & (df_existing["declination (deg)"] == self.declination)
        if match.any():
            print("Overwritting previous entry...")
            df_existing = df_existing[df_existing["right_ascension (deg)"] != self.right_ascension]

        print(self.get_subtraction_used())

        # Creates new entry
        new_instance_data = {
            "right_ascension (deg)"         : self.right_ascension,
            "declination (deg)"             : self.declination,
            "best_angle (deg)"              : self.best_angle,
            "radius (arcmin)"               : self.radius_list[-1] * self.pixel_width_arcmins,
            "number_of_increases"           : len(self.light_values_2d)-1,
            "uncertainty_best_angle (deg)"  : self.uncertainty,
            "nan_percentage (%)"            : self.nan_percentage,
            "noise_percentage (%)"          : self._calculate_noise(),
            "adjustment_status"             : self.adjustment_status,
            "redshift"                      : redshift,
            "redshift_spectroscopic_bool"   : are_Spectroscopic_bool,
            "subtraction_used"              : self.get_subtraction_used(),
            "length_angular_means"          : self.length_angular_means[self._find_catalogue_match_index(self.right_ascension, self.declination)],
        }

        df_new = pd.DataFrame([new_instance_data]).astype(df_existing.dtypes.to_dict())

        # Updates Excel sheet
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        df_updated.to_excel(self.excel_file, index=False)

    def add_text_to_ax(self, ax, text):
        ax.text(
            self.image.shape[0] * 0.05, self.image.shape[1] * 0.88,
            text,
            fontsize = self.axis_font_size,
            color = "white",
            bbox = dict(facecolor = "gray", edgecolor = "none", boxstyle = "round,pad=0.3", alpha = 0.5)
        )


    def _save_plots(self):
        """
        Saves plots based on self.folder_status.
        """

        save_location_mapping = {
            "auto"      : self.save_auto_loc,
            "manual"       : self.save_manual_loc,
            "some_manual"  : self.save_some_manual_loc,

        }

        for status, save_location in save_location_mapping.items():
            if self.adjustment_status == status :

                num_of_rows = len(self.light_values_2d)
                fig = plt.figure(figsize=(11, 4 * num_of_rows))

                gs = fig.add_gridspec(len(self.light_values_2d), 3, width_ratios = [1, .1, 1.2], hspace = .01)

                for i in range(len(self.light_values_2d)):



                    scatter_axis = fig.add_subplot(gs[i, 2])
                    overlay_axis = fig.add_subplot(gs[i, 0])

                    self.create_scatter_ax(scatter_axis, self.light_values_2d[i], self.light_values_convolved_2d[i], self.threshold_list[i], self.best_angle_index_list[i])
                    self.create_overlay_ax(overlay_axis, self.best_angle_index_list[i], self.radius_list[i])

                    scatter_axis.get_xaxis().set_visible(False)
                    overlay_axis.get_xaxis().set_visible(False)


                    if i == 0:

                        v_min       = np.nanpercentile(self.image, 10)
                        v_max       = np.nanpercentile(self.image, 99.9)
                        object_norm = ImageNormalize(self.image, interval=ManualInterval(vmin=v_min, vmax=v_max), stretch=SqrtStretch())


                        # Draw subplot components
                        h, w = self.image.shape
                        crop_img = self.image[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
                        flipped_img = np.flipud(crop_img)

                        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                        inset_ax = inset_axes(overlay_axis, width = "30%", height = "30%", loc = "upper right", borderpad = 0.2 )

                        # inset_ax = overlay_figure.add_axes([0.7, 0.65, 0.3, 0.3])
                        # inset_ax.axis("off")

                        center_x, center_y = (flipped_img.shape[1]-1) / 2, (flipped_img.shape[0]-1) / 2

                        inset_ax.imshow(flipped_img, norm=object_norm, cmap = "inferno")

                        for side in ['top', 'bottom', 'left', 'right']:
                            inset_ax.spines[side].set_visible(True)
                            inset_ax.spines[side].set_edgecolor("grey")
                            inset_ax.spines[side].set_linewidth(1)
                            inset_ax.spines[side].set_zorder(10)  # Bring border above image

                        inset_ax.set_xticks([])
                        inset_ax.set_yticks([])

                        circle = patches.Circle((center_x, center_y), self.image.shape[0]*.01, facecolor="red", linewidth = 1, edgecolor="orange", fill=True, )
                        inset_ax.add_patch(circle)


                    if i == len(self.light_values_2d) - 1:

                        if self.adjustment_status == "manual":
                            overlay_axis.plot(self.manual_x, self.manual_y, color = "lightgreen", linewidth=2)
                            scatter_axis.axvline(self.best_angle, color = "green", linewidth = 1, linestyle = "--")
                            self.add_text_to_ax(overlay_axis, "Manual")
                        elif self.adjustment_status == "some_manual":
                            num = int(self.radius_list[i]) // self.increase_radius_step
                            if num == 1 and self.radius_list[0] == self.radius_list[1]:
                                self.add_text_to_ax(overlay_axis, "Selected: 0")
                            else:
                                self.add_text_to_ax(overlay_axis, "Selected: " + str(num))

                        else:
                            self.add_text_to_ax(overlay_axis, int(self.radius_list[i]) // self.increase_radius_step)

                        scatter_axis.get_xaxis().set_visible(True)
                        overlay_axis.get_xaxis().set_visible(True)

                    else:
                        self.add_text_to_ax(overlay_axis, i)

                

                fig.savefig(save_location, format = self.save_format, bbox_inches = "tight")
                # plt.show()
                plt.close()


    # --- Helper ---

    def _find_catalogue_match_index(self, ra, dec):
        """
        """

        number_of_jet_systems = self.right_ascensions.shape[0]

        for i in range(number_of_jet_systems):
            if (float('{:.6f}'.format(self.right_ascensions[i])) == float(ra)
                    and float('{:.6f}'.format(self.declinations[i])) == float(dec)):
                return i

    def _check_adjustment_status(self):
        """
        Check what adjustment status is being applied based on the manually inputted lists:
        self.move_to_good_list, self.increase_list, self.decrease_list, self.exclude_list, self.manual_adjustment_list
        Sets: self.adjustment_status
        """

        status_mapping = {
            "manual"           : self.manual_list,
            "some_manual"      : self.some_manual_list,
        }

        for status, coord_list in status_mapping.items():
            for entry in coord_list:
                right_ascension, declination = map(float, entry[0].split("_"))

                if self.right_ascension == right_ascension and self.declination == declination:
                    self.adjustment_status = status
                    print("Check adjustment status:", self.adjustment_status)
                    break



    def _find_peaks(self, light_values_convolved):
        """
        Returns the indices of the peak values in self.light_values_convolved
        """
        values = light_values_convolved
        extended = np.concatenate((values[-10:], values, values[:10]))
        peaks, _ = find_peaks(extended)
        peaks = peaks - 10
        peaks = peaks[(peaks >= 0) & (peaks < len(values))]
        return peaks


    def _create_coordinate_list(self, angle, radius):
        """
        Creates and returns a list of x values and y values for a line based on the angle given.
        """

        x_values = list()
        y_values = list()

        for r in range(-1 * radius, radius + 1):

            x_val = self.central_point[1] + (-1) * r * math.sin(angle)
            y_val = self.central_point[0] + r * math.cos(angle)

            x_values.append(x_val)
            y_values.append(y_val)

        return [x_values, y_values]

    def _find_uncertainty(self, threshold):
        """
        Finds the distance from self.best_angle to the furthest peak above the threshold provided. Note: this graph's
        behavior is cyclic, where 0 degrees and 180 degrees are the same.
        """

        # Index of each peak
        maxima_indices      = self._find_peaks(self.light_values_convolved_2d[-1])
        # Coordinates for each peak
        x_values, y_values  = ((self.angles * 180 / np.pi)[maxima_indices],
                              self.light_values_convolved_2d[-1][maxima_indices])
        # Calculating perpendicular based on self.best_angle
        if self.best_angle + 90 > 180:
            perpendicular = self.best_angle - 180 + 90
        else:
            perpendicular = self.best_angle + 90

        # Split the peaks that are above the threshold into two regions:
        # - Region 1 (all peaks above threshold from perpendicular to self.best_angle)
        # - Region 2 (all peaks above threshold from self.best_angle to perpendicular)
        # Calculate uncertainty based on position
        uncertainties = []
        for j in list(range(0, len(x_values))):
            # print(threshold, y_values[j])
            if y_values[j] > threshold:
                if perpendicular < self.best_angle:
                    # If current peak is in region 2 and less than perpendicular
                    if x_values[j] < perpendicular:
                        uncertainties.append(180 - self.best_angle + x_values[j])
                    # If current peak is region 2 and greater than perpendicular
                    elif x_values[j] >= self.best_angle:
                        uncertainties.append(x_values[j] - self.best_angle)
                    # If current peak is region 1
                    elif perpendicular <= x_values[j] < self.best_angle:
                        uncertainties.append(self.best_angle - x_values[j])
                # print(uncertainties)

                if perpendicular > self.best_angle:
                    # If current peak is in region 1 and less than self.best_angle
                    if x_values[j] < self.best_angle:
                        uncertainties.append(self.best_angle - x_values[j])
                    # If current peak is in region 1 and greater than perpendicular
                    elif x_values[j] >= perpendicular:
                        uncertainties.append(180 - x_values[j] + self.best_angle)
                    # If current peak is in region 2 and less than perpendicular
                    elif self.best_angle <= x_values[j] < perpendicular:
                        uncertainties.append(x_values[j] - self.best_angle)
        uncertainty = max(uncertainties)

        return int(uncertainty)

    def _adjust_radius(self, radius):
        """
        Adjusts radius to the provided radius value and regenerates the brightest angle.
        """

        self.radius_list.append(radius)
        self._find_brightest()

    def _calculate_noise(self):
        """
        Calculates the distance from each point in self.light_values to each point in self.light_values_convolved.
        """

        actual_points       = np.array(self.light_values_2d[-1])
        smoothed_points     = np.array(self.light_values_convolved_2d[-1])

        # Use smoothed points to calculate range so that calculation is not affected by outliers
        light_value_range   = max(smoothed_points) - min(smoothed_points)
        noise_percentage    = 0
        if light_value_range != 0:
            differences         = np.abs(actual_points - smoothed_points)/light_value_range
            noise_percentage    = np.mean(differences) * 100

        return noise_percentage

    def _make_adjustments(self):
        """
        Uses self.adjustment status to determine what adjustments to make and what folder to save the files to.
        """


        if self.adjustment_status == "manual":
            entry = [item for item in self.manual_list if
                     (f"{self.right_ascension:.6f}_{self.declination:.6f}") in item]
            self.best_angle = entry[0][1]
            self.radius_list.append(int(self.jet_width_pixels / 4))
            self.manual_x, self.manual_y = self._create_coordinate_list(np.radians(self.best_angle), self.radius_list[-1])

        elif self.adjustment_status == "some_manual":
            entry = [item for item in self.some_manual_list if
                     (f"{self.right_ascension:.6f}_{self.declination:.6f}") in item]
            radius_increase_num = entry[0][1]

            if radius_increase_num > (self.jet_width_pixels / 2 / self.increase_radius_step):

                raise ValueError(
                    "WARNING -Some_manual entry for "
                        + str(self.right_ascension) + " + " + str(self.declination) +
                    " must be between 0 and " + str(int(self.jet_width_pixels / 2 / self.increase_radius_step))
                )

            adjusted_radius = radius_increase_num * self.increase_radius_step
            if adjusted_radius == 0:
                adjusted_radius = self.starting_radius
            if adjusted_radius < self.starting_radius:
                self._adjust_radius(self.starting_radius)
            else:
                self._adjust_radius(adjusted_radius)
            self.uncertainty    = self._find_uncertainty(self.threshold_list[-1])


    def create_scatter_ax(self, scatter_ax, light_values, light_values_convolved, threshold, best_angle_index):
        """
        Creates a scatter plot of the data and saves it to self.scatter_plot_data
        """

        # === Formating ===
        scatter_ax.set_ylabel(r'mean SI $I_\nu$' + '\n' r'$(\mathrm{mJy\ arcmin^{-2}})$', fontsize = self.axis_font_size)
        scatter_ax.set_xlabel(r"line segment PA ($\degree$)", fontsize = self.axis_font_size)



        # === Plot data ===
        # --- Plot light values data ---
        angles_degrees      = self.angles * 180 / np.pi
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
        best_angle = np.degrees(self.angles[best_angle_index])
        scatter_ax.axvline(x = best_angle, color = "cadetblue", linewidth = 1, linestyle = "--")
        scatter_ax.axvline(x = best_angle, color = "cadetblue", linewidth = 1, linestyle = "--")

        # --- Mark absolute and local maxima ---
        maxima_indices = self._find_peaks(light_values_convolved)

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
        # print(yticks)
        # print([f"{i * 1000:.0f}" for i in yticks])

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

    def create_overlay_ax(self, overlay_ax, best_angle_index, radius):
        """
        Creates a plot of the cutout image and overlays the line of the best angle. Saves to self.overlay_image_data.
        """

        # Formating
        dpi = 1000

        # Calculate v_min and v_max

        v_min       = np.nanpercentile(self.image, 10)
        v_max       = np.nanpercentile(self.image, 99.9)
        object_norm = ImageNormalize(self.image, interval=ManualInterval(vmin=v_min, vmax=v_max), stretch=SqrtStretch())

        # overlay_figure, overlay_ax = plt.subplots(figsize=(7.3, 6))

        overlay_ax.set_xlim(0, self.image.shape[1])
        overlay_ax.set_ylim(0, self.image.shape[0])
        overlay_ax.set_xlabel(r"right ascension ($\degree$)", fontsize = self.axis_font_size)
        overlay_ax.set_ylabel(r"declination ($\degree$)", fontsize = self.axis_font_size)

        overlay_ax.patch.set_facecolor('purple')

        # overlay_figure.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.12)

        # Converting ax units
        pixel_size_arcmin = self.calculate_pixel_size_deg(self.image, self.image_width_arcmins) # in deg

        shift = self.format_ticks(self.image, pixel_size_arcmin, self.declination, overlay_ax, 1, 0, self.declination)
        self.format_ticks(self.image, pixel_size_arcmin, self.right_ascension, overlay_ax, 0, shift, self.declination)

        # Draw main plot components
        overlay_ax.imshow(self.image, origin="lower", cmap="inferno", norm=object_norm)

        x, y            = self._create_coordinate_list(self.angles[best_angle_index], radius)
        overlay_ax.plot(x, y, color="cyan", linewidth=2)

        circle          = plt.Circle(xy=(self.central_point[0], self.central_point[1]), radius = radius, fill = True,
                               facecolor = "white",
                               edgecolor = "green", alpha=0.3)
        central_point   = plt.Circle(xy=(self.central_point[0], self.central_point[1]), radius=self.image.shape[0]*.01, fill=True,
                                      facecolor="red", zorder=10)
        overlay_ax.add_artist(circle)
        overlay_ax.add_artist(central_point)