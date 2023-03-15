import visa
from time import sleep
import ThorlabsPM100 as tl
from  frghardware.components import mono
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymeasure.instruments.srs import SR830
%matplotlib

class EQE:
    """
    This code is for carrying out EQE measurements in SERF156
    """
    def __init__(self) -> None:

        self.connect()
        self.set_wls()
        self.lightsource_correction()
        self.data = {}
        self.lia.time_constant = 0.3    #shorter time-constants could be used, but 300ms gives great data
        print('Ready to take EQE.')
        self.m.wavelength = 532         # make the light visible to align sample

    def connect(self):
    
        """
        This code connects to and configures the following instruments for measurement:
        Lockin amp, calibrated photodiode, and the monochromator.
        """

        self.lia = SR830('GPIB0::8::INSTR')
        input("Lock-in amplifier object initiated.\nPress Enter to continue...")

        self.m = mono.Mono()
        input("Mono object initiated. \nPress Enter to continue... \n")
        self.m.open_shutter()
        input("Shutter open.\nPress Enter to continue... \n")
        self.m.wavelength = 532
        input("Wavelength set to 532nm.\nPress Enter to continue... \n")

        rm = visa.ResourceManager()
        input("Resource manager initiated.\nPress Enter to continue... \n")
        inst = rm.open_resource('USB0::0x1313::0x8072::P2006516::INSTR', timeout = 10000)
        self.pm = tl.ThorlabsPM100(inst=inst)
        input("PM100 (Power Meter) object initiated.\nPress Enter to continue... \n")
        self.pm.configure.scalar.power()
        input("PM100 (Power Meter) configured for power.\nPress Enter to continue... \n")

    def set_wls(self):
        """
        Creates a np.array of the wavelengths deisred and assigns them to the wls attribute
        """
        start = int(input("Start Wavelength = "))
        stop = int(input("Stop Wavelength = "))
        step = int(input("Interval = "))
         
        self.wls = np.arange(start, stop+step, step)

        print(f'\nThe wavelengths that will be used are:\n{self.wls}')

    def lightsource_correction(self, sample_name = 'Lamp_Power.csv'):
        """
        Take readings of the power hitting the sample using the calibrated photodiode.
        """
        print('\nSetting wavelength to 532 nm for ease of visibility...\n')
        self.m.wavelength = 532

        input('Please place calibrated Si photodiode centered on spot. \nPress Enter when in place \n')
        input('Please ensure chopper wheel is turned on and beam is unobstructed. \nPress Enter when done \n')
        input('Eliminate stray light in room. \nPress Enter to begin the lightsource correction \n')

        N_AVG = 20     # number of times to query the photodiode
        self.pm.sense.average.count = 200    # number of times the photodiode interally averages readings per external query
                                             # each reading is approx. 3ms, so 200 readings ~600ms

        pm_power = []
        pm_stds = []

        self.m.wavelength = self.wls[0]
        self.m.open_shutter()

        for wl in tqdm(self.wls, desc = 'Measuring lamp power'):
            self.m.wavelength = wl
            self.pm.sense.correction.wavelength = wl
            sleep(0.1)
            temp = []
            for i in range(N_AVG):
                temp.append(self.pm.read)

            temp = np.array(temp)

            pm_power.append(temp.mean())
            pm_stds.append(temp.std())

        df = pd.DataFrame(
            {
                'wavelength': self.wls,
                'pm_power': pm_power,
                'pm_stds' : pm_stds
            }
        )
        
        self.source_correction = df

        df.to_csv(sample_name, index = False)

    def take_EQE(self, sample_name):

        """
        The function that talks to everything and takes EQE
        """

        N_AVERAGES = 20    # number of times to query the lockin

        lia_voltages = []
        lia_voltage_stds = []

        for _ in range(5):
            waste = self.lia.time_constant

        tc = self.lia.time_constant

        for wl in tqdm(self.wls, desc = 'Taking EQE'):
            temp_lia = []
            self.m.wavelength = wl     # change the wavelength on the mono

            sleep(15*tc) # let the lockin...uh...lock in.
            for _ in range(5):
                waste = self.lia.magnitude
            for i in range(N_AVERAGES):   #average over 20 queries
                temp_lia.append(self.lia.magnitude)
                sleep(tc)  # put one tc between the queries. this is an attempt to avoid sampling faster than the output updates

            lia_voltages.append(np.mean(temp_lia))
            lia_voltage_stds.append(np.std(temp_lia))

        print('Resetting wavelength to 532 nm for visibility...')
        self.m.wavelength = 532

        energies, qe, qe_error = self._calc_eqe(lia_voltages, lia_voltage_stds)

        df = pd.DataFrame(
            {
                'wls': self.wls,
                'ev' : 1239.84193/self.wls,
                'lia_voltages': lia_voltages,
                'lia_voltage_stds': lia_voltage_stds,
                'eqe' : qe,
                'eqe_error' : qe_error
            }
        )

        df.to_csv(sample_name, index = False)

        plt.figure(figsize=(4, 3), dpi=360)
        plt.plot(1239.84193 / energies, qe*100, label="EQE", linewidth=0.7, color="black")
        plt.fill_between(1239.84193 / energies, (qe-qe_error)*100, (qe+qe_error)*100, color = 'lightgrey')
        plt.xlim(min(self.wls), max(self.wls))
        plt.ylim(-10, 110)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("EQE (%)")

        self.data[sample_name] = df
        df.to_csv(sample_name, index = False)

    def _get_n_photons(self, wavelength, power):
        # get the number of photons from the power meter
        hc = 1.98644568e-25  # in joules*meters
        photon_energy = hc / (wavelength * 1e-9)  # wavelength in nm converted to meters
        n_photons = power / photon_energy
        return n_photons

    def _nm_to_eV(self, wavelength):
        # convert wavelengths to eV
        hc = 1239.84193  # in eV*nm
        eV = hc / np.array(wavelength)  # wavelength in nm converted to meters
        return eV

    def _get_n_electrons(self, voltage, resistance=50):
        # get number of electrons from the sample being measured with lock-in amplifier
        # the voltage is in volts and the resistance is in ohms
        #  50 ohms is the default resistance of the lock-in amplifier + shunt resistor
        q = 1.60217662e-19  # in coulombs
        n_electrons = (voltage / resistance) / q
        return n_electrons

    def _error_fraction(self, signal, error):
        # get the percent error of the signal
        error_frac = error / signal
        return error_frac

    def _get_eqe_error(self, n_electrons_error, power_error):
        # get the error of the EQE
        eqe_error = np.sqrt((n_electrons_error) ** 2 + (power_error) ** 2)
        return eqe_error

    def _calc_eqe(self, lia_v, lia_std):

        wls = self.wls
        energies = self._nm_to_eV(wls)

        ref_powers = self.source_correction['pm_power']
        ref_stds = self.source_correction['pm_stds']

        n_photons = self._get_n_photons(wls, ref_powers)
        n_photons_error = self._error_fraction(ref_powers, ref_stds)

        n_electrons = self._get_n_electrons(np.array(lia_v))
        n_electrons_error = self._error_fraction(np.array(lia_v), np.array(lia_std))

        eqe = n_electrons / n_photons
        eqe_error = eqe * self._get_eqe_error(n_electrons_error, n_photons_error)
        return energies, eqe, eqe_error
