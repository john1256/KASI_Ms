import numpy as np
import scipy.integrate as spi

class EisensteinHu:
    def __init__(self, **kwargs):
        self.name = "Eisenstein-Hu"
        default_params = {
            'ombh2': 0.02242,
            'omch2': 0.11933,
            'omlambh2' : 0.6889,
            'omRh2': 0.0,
            'h': 0.67,
            'mnu': 0.0,
            'ns': 0.965,
            'As': 2e-9,
            'tau': 0.0632,
            'TCMB': 2.725
        }
        default_params.update(kwargs)
        self.__dict__.update(default_params)
        self.params = default_params

        self.om0h2 = self.ombh2 + self.omch2
        self.om0 = self.om0h2 / self.h**2
        self.omlamb = self.omlambh2 / self.h**2
        self.omb = self.ombh2 / self.h**2
        self.omc = self.omch2 / self.h**2
        self.f_baryon = self.omb/self.om0
        self.theta = self.TCMB/2.7
        self.zeq = 2.5*1e4 * self.om0h2 * (self.theta)**-4
        self.keq = 7.46*1e-2 * self.om0h2*(self.theta)**-2 # in Mpc^-1

        # k_silk calculation
        self.ksilk = 1.6 * self.ombh2**0.52 * self.om0h2**0.73 * ( 1+ (10.4*self.om0h2) ** -0.95 ) # in Mpc^-1

    # calculate zdrag [4]

        self.z_drag_b1 = 0.313 * self.om0h2 ** -0.419 * (1 + 0.607 * self.om0h2 ** 0.674)
        self.z_drag_b2 = 0.238 * self.om0h2 ** 0.223
        self.zdrag    = 1291 * self.om0h2 ** 0.251 / (1. + 0.659 * self.om0h2 ** 0.828) * \
                           (1. + self.z_drag_b1 * self.ombh2 ** self.z_drag_b2) # [4]

    # calculate s (sound horizon)    [6]
        self.Req = 31.5*self.ombh2*(self.theta)**-4 * (1e3/self.zeq)
        self.Rdrag = 31.5*self.ombh2*(self.theta)**-4 * (1e3/self.zdrag)
        self.s = 2. / (3.*self.keq) * np.sqrt(6. / self.Req) * \
                    np.log((np.sqrt(1 + self.Rdrag) + np.sqrt(self.Rdrag + self.Req)) / (1 + np.sqrt(self.Req)) )
    # Calculate alpha_c, beta_c [11],[12]
        a1 = (46.9*self.om0h2)**0.670*(1+(32.1*self.om0h2)**-0.532)
        a2 = (12.0*self.om0h2)**0.424*(1+(45.0*self.om0h2)**-0.582)
        self.alpha_c = a1**(-self.f_baryon)*a2**(-self.f_baryon**3) # [11]
        
        b1 = 0.944 / (1 + (458*self.om0h2) ** -0.708)
        b2 =  0.395 * self.om0h2 ** -0.0266
        self.beta_c = 1. / ( 1 + b1 * ((1-self.f_baryon) ** b2 - 1) ) # [12]

    # Calculate alpha_b [14]
        def G(y): # [15]
                return y * ( (-6. * np.sqrt(1 + y)) + ( (2 + 3 * y) * np.log( (np.sqrt(1+y) + 1) / (np.sqrt(1+y)-1) ) ) )
        self.alpha_b = 2.07*self.keq*self.s*(1+self.Rdrag)**-(3/4)*G((1+self.zeq)/(1+self.zdrag)) # [14]    
        self.beta_node = 8.41*self.om0h2**0.435 # [23]
        self.beta_b = 0.5 + self.f_baryon + ( 3 - 2*self.f_baryon )*np.sqrt( (17.2*self.om0h2)**2 + 1) # [24]

    def print_params(self):
        for key, value in self.params.items():
            print(f"{key}: {value}")
        return self.params
    
    def Transfer_function_small_scale(self,k):
        q = k/(13.41*self.keq) # [10]
        ks = k*self.s
        T_c = self.alpha_c * np.log(18*self.beta_c*q) / (14.2*q**2) # [9]
        T_b = self.alpha_b * np.sin(ks)/ks
        return (1-self.f_baryon)*T_c + self.f_baryon*T_b

    def Transfer_function(self, k):
        # Calculate T_c
        q = k/(13.41*self.keq) # [10]
        ks = k*self.s
        def C(alpha_c, q):  # [20]
            return 14.2/alpha_c + 386./(1+69.9*q**1.08)
        def Tilde0(q, alpha_c, beta_c): #  [19]
            return np.log(np.e+1.8*beta_c*q)/(np.log(np.e+1.8*beta_c*q)+C(alpha_c, q)*q**2)

        f = 1/(1+(ks/5.4)**4) # [18]
        T_c = f*Tilde0(q, 1, self.beta_c) + (1-f)*Tilde0(q, self.alpha_c, self.beta_c) # [17]

        # Calculate T_b
        self.stildek = self.s/((1+ (self.beta_node/ks)**3 )**(1/3)) # [22]
        def j0(x):
            return np.sin(x)/x
        T_b = ( Tilde0(q,1,1)/(1+(ks/5.2)**2) + self.alpha_b/(1+(self.beta_b/ks)**3) * np.exp(-(k/self.ksilk)**1.4) )  * j0(k*self.stildek) # [19]
        # Total Transfer function
        T_k = self.f_baryon*T_b + (1-self.f_baryon)*T_c # [16]
        return T_k
    def first_peak(self):
        s = 44.5 * np.log(9.83 / self.om0h2) / np.sqrt(1 + 10 * self.ombh2**3/4) # in Mpc [26]
        k_peak = 5* np.pi / (2*s) * (1 + 0.217 * self.om0h2) # in Mpc^-1 [25]
        return k_peak
    def Transfer_function_zero_baryon(self, k):
        k_in_hMpc = k * self.h
        self.gamma = self.om0h2
        q = k_in_hMpc * self.theta**2 / self.gamma # use k in h Mpc^-1
        def L0(q): # [29]
            return np.log(2*np.e + 1.8*q)
        def C0(q): # [30]
            return 14.2 + 731./(1+62.5*q)
        T0 = L0(q)/ (L0(q) + C0(q)*q**2) # [28]
        return T0
    
    def Transfer_function_nowiggles(self, k):
        ks = k*self.s
        alpha_gamma = 1 - 0.328 * np.log(431*self.om0h2) * self.f_baryon + 0.38 * np.log(22.3*self.om0h2) * self.f_baryon**2 # [31]
        gamma_eff = self.om0h2/self.h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks)**4)) # [30]
        k_in_hMpc = k * self.h
        q = k_in_hMpc * self.theta**2 / gamma_eff
        def L0(q): # [29]
            return np.log(2*np.e + 1.8*q)
        def C0(q): # [30]
            return 14.2 + 731./(1+62.5*q)
        T0 = L0(q)/ (L0(q) + C0(q)*q**2) # [29]
        return T0


    def Growth_factor(self, z):
        def Omega_m_z(z):
            return self.om0h2 * (1+z)**3 / (self.omlambh2 + self.omRh2*(1+z)**2 + self.om0h2*(1+z)**3) # [A5]
        def Omega_lamb_z(z):
            return self.omlambh2 / (self.omlambh2 + self.omRh2*(1+z)**2 + self.om0h2*(1+z)**3) # [A6]
        def D_1(z): # [A4]
            Omz = Omega_m_z(z)
            Olambz = Omega_lamb_z(z)
            return (5*Omz/2) / ( Omz**(4/7) - Olambz + (1 + Omz/2)*(1 + Olambz/70) ) / (1+z)
        return(D_1(z))
    def Power_spectrum_0(self,k):
        A = self.As * (k/0.05)**(self.ns - 1)
        T_k = self.Transfer_function(k)
        self.speed_of_light = 299792.458 # in km/s
        H0 = self.h * 100 # in km/s/Mpc
        ntilde = self.ns-1
        if self.omlambh2 == 0:
            delta_H = 1.95 * 1e-5 * self.om0**(-0.35-0.19*np.log(self.om0)-0.17*ntilde) * np.exp(-ntilde-0.14*ntilde**2) # [A2]
            
        else:
            delta_H = 1.94 * 1e-5 * self.om0**(-0.785-0.05*np.log(self.om0)) * np.exp(-0.95*ntilde-0.169*ntilde**2) # [A2]
        P_k = 2*np.pi**2/k**3 * delta_H**2 * (self.speed_of_light*k/H0)**(3+ self.ns) * T_k**2
        return P_k
    def Power_spectrum_z(self,k,z):
        P_k_0 = self.Power_spectrum_0(k)
        D1_z = self.Growth_factor(z)
        return P_k_0 * D1_z**2
    def Power_spectrum_dimensionless(self,P_k_0,k):
        """
        Input : P_k_0 : Power spectrum at z=0
        """
        return k**3 * P_k_0 / (2*np.pi**2)
    def Sigma_R2(self,P_k_dimless,karr, R):
        """
        Input : P_k_dimless : Dimensionless power spectrum at z=0
        """
        def j1(x):
            return (x*np.cos(x) - np.sin(x))/x**2
        def integrand(k):
            x = k*R
            W = 3*j1(x)/x
            return 1/k * P_k_dimless * W**2
        integral = np.trapz(integrand(karr), karr) # [A7]
        return integral
    def Power_spectrum_liddle(self, a, k, Transfer = 'EisensteinHu', rescale = False):
        """ Using a definition from Liddle & Lyth Cosmological Inflation and Large-Scale Structure (2000)
        """
        P_primodial = self.As * (k/0.05)**(self.ns - 1)
        H0 = self.h * 100 # in km/s/Mpc
        if Transfer == 'EisensteinHu':
            T_k = self.Transfer_function(k)
        elif Transfer == 'ZeroBaryon':
            T_k = self.Transfer_function_zero_baryon(k)
        elif Transfer == 'NoWiggles':
            T_k = self.Transfer_function_nowiggles(k)
        Hubble = H0 * np.sqrt(self.om0/a**3 + self.omlamb)
        D1 = 5/2 * self.om0 * H0**2 * Hubble * spi.quad(lambda a: 1/(a*Hubble)**3,0,a)[0] # [6.10]
        #D1 = 5/2 * (1/70 + 209*self.om0/140 - self.om0**2/140 + self.om0**(4/7))**-1 / a # [6.12]
        P_k = 4/25 * k * self.speed_of_light**4 / (2*np.pi**2) / (a*Hubble)**4 * P_primodial * T_k**2 * D1**2/a**2 # [6.14]

        if rescale == True:
            P_k_dimless = self.Power_spectrum_dimensionless(P_k, k)
            
            sigma8 = self.Sigma_R2(P_k_dimless, k, 8)
            print("Rescaling power spectrum... sigma8 =", sigma8**0.5)
            True_sigma82 = 0.811**2
            P_k = P_k * True_sigma82/sigma8

        return P_k 