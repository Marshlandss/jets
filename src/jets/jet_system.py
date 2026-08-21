
# === imports ===

import math
import os
import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve

from scipy.signal import find_peaks

# === JetSystem class ===

class JetSystem:

    def __init__(self, path, ra, dec, image_width_arcmins, adjustment_status, sa_list, m_list,
                 angles, right_ascensions, declinations, length_angular_means, redshifts, are_Spectroscopic_bool_list,
                 percentage, nan_percentage_cutoff,
                 excel_file, pixel_shift):

        # === Attributes ===
        # --- External ---
        self.right_ascensions               = right_ascensions
        self.declinations                   = declinations
        self.length_angular_means           = length_angular_means
        self.redshifts                      = redshifts
        self.angles                         = angles
        self.are_Spectroscopic_bool_list    = are_Spectroscopic_bool_list
        self.sa_list                        = sa_list
        self.m_list                         = m_list

        # --- Initializing ---
        self.path                       = path
        self.right_ascension            = ra
        self.declination                = dec
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
        self.adjustment_status          = adjustment_status
        self.manual_x                   = []
        self.manual_y                   = []
        self.increase_radius_step       = 0
        self.image_width_arcmins        = image_width_arcmins

        hdu_list                        = fits.open(self.path)
        self.image                      = hdu_list[0].data
        self.image                      = self.image.copy()/(self.beamGaussian(6,6)*3600)

        self.image                      = np.roll(np.roll(self.image, -pixel_shift, axis=1), -pixel_shift, axis=0)
        hdu_list.close()

        self.image_side_length          = self.image.shape[0]
        self.central_point              = [(self.image_side_length - 1) / 2., (self.image_side_length - 1) / 2.]
        self.nan_percentage             = (np.sum(np.isnan(self.image)) / (self.image.shape[0] * self.image.shape[1])) * 100

        self.pixel_width_arcmins        = self.image_width_arcmins / self.image_side_length

        self.excel_file                 = excel_file

        # === Function Calls ===
        print("> Creating new files...")

        index                           = self._find_catalogue_match_index(self.right_ascension, self.declination)
        length_angular_mean             = self.length_angular_means[index]
        redshift                        = self.redshifts[index]
        are_Spectroscopic_bool          = self.are_Spectroscopic_bool_list[index]
        resolution_arcsec               = 6

        if length_angular_mean == 0 or length_angular_mean == None:
            print("!!! SKIPPED - No match for jet system:", self.right_ascension, "+", self.declination)

        else :

            self.jet_width_pixels       = length_angular_mean / self.pixel_width_arcmins # in pixels
            self.increase_radius_step   = int((self.jet_width_pixels / 2) /7)
            if (length_angular_mean * 60)/2 * 0.1 < 6:
                print("\n> Starting radius too small; increasing to FWHM.")
                self.starting_radius = int((resolution_arcsec / 60) / self.pixel_width_arcmins) # in pixels
            else:
                print("\n> Starting radius good.")
                self.starting_radius        = int(self.jet_width_pixels / 20) # in pixels

            self.radius_list.append(self.starting_radius)
            self._find_brightest()

            if self.nan_percentage > self.nan_percentage_cutoff:
                print("!!! WARNING - ", self.nan_percentage, "% of light values are NaN:", self.right_ascension, "+", self.declination)

            self._find_best_radius()
            print("\n>", self.right_ascension, "+", self.declination)
            print("> Jet angle: ", self.best_angle)
            print("> Status: ", self.adjustment_status)
            print("> Redshift: ", redshift)
            print("> Subtraction used: ", self.get_subtraction_used())

            self._record_jet_data(redshift, are_Spectroscopic_bool)

    # === Functions ===
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
            print("\n> Overwritting previous excel entry...")
            df_existing = df_existing[df_existing["right_ascension (deg)"] != self.right_ascension]

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

    # --- Helper ---

    def _find_catalogue_match_index(self, ra, dec):
        """
        """

        number_of_jet_systems = self.right_ascensions.shape[0]

        for i in range(number_of_jet_systems):
            if (float('{:.6f}'.format(self.right_ascensions[i])) == float(ra)
                    and float('{:.6f}'.format(self.declinations[i])) == float(dec)):
                return i

    def find_peaks(self, light_values_convolved):
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
        maxima_indices      = self.find_peaks(self.light_values_convolved_2d[-1])
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


        if self.adjustment_status == "m":
            entry = [item for item in self.m_list if
                     (f"{self.right_ascension:.6f}_{self.declination:.6f}") in item]
            self.best_angle = entry[0][1]
            self.radius_list.append(int(self.jet_width_pixels / 4))
            self.manual_x, self.manual_y = self._create_coordinate_list(np.radians(self.best_angle), self.radius_list[-1])

        elif self.adjustment_status == "sa":
            entry = [item for item in self.sa_list if
                     (f"{self.right_ascension:.6f}_{self.declination:.6f}") in item]
            radius_increase_num = entry[0][1]

            if radius_increase_num > (self.jet_width_pixels / 2 / self.increase_radius_step):

                raise ValueError(
                    "WARNING -SA entry for "
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


    

    