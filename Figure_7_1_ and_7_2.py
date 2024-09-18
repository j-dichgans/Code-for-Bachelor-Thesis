import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
import tqdm
import scipy.signal
import statsmodels.tsa.stattools


#define time interval, T should be integer
T = 10000
window_length = 200 



########################################################

def mu(t):
    return 2/T * t - 1

def f(x,t): 
    return (x - x**3 /3.0 - mu(t))


def get_equilibria_paths():
    n = T + 1
    ts = np.linspace(0,T,n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    unstable = np.zeros(n)

    for i in tqdm.trange(n):
        sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,50),[2.0], method="BDF")
        upper[i] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,50),[-2.0], method="BDF")
        lower[i] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x: -f(x,ts[i]),(0.0,50),[0.0], method="BDF")
        unstable[i] = sol.y[0,-1]
     
    unstable[np.abs(unstable)>1] = np.nan

    return ts,upper,lower,unstable



def get_var(x):
    n_windows = int(T/window_length) 
    var = np.full(n_windows,np.nan)
    for i in tqdm.trange(n_windows):
        var[i] = statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2).var()
    return var



def get_ar(x):
    n_windows = int(T/window_length) 
    ar = np.full(n_windows,np.nan)
    for i in tqdm.trange(n_windows):
        ar[i] = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2))[1]
    return ar



def get_ls(x,noise,a):

    def fitfunction(f,ls):
        return np.log(1/(f**2  + ls**2))
    

    n_windows = int(T/window_length) 
    ls = np.full(n_windows,np.nan)

    for i in tqdm.trange(n_windows):
        frequencies = 2*np.pi*(1/window_length)*np.arange(1,window_length/2)                                                

        xs_window_detrend = statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2)
        noise_window = noise[i*window_length:(i+1)*window_length]
        
        estim_psd_xs_wn = np.array([np.abs(1/np.sqrt(window_length)*(np.exp(-1j*frequencies[j]*np.arange(0,window_length)) @ xs_window_detrend))**2 for j in range(int(window_length/2) - 1)])
        estim_psd_xi_wn = np.array([np.abs(1/np.sqrt(window_length)*(np.exp(-1j*frequencies[i]*np.arange(0,window_length)) @ (noise_window*a)))**2 for i in range(int(window_length/2) - 1)])

        popt = scipy.optimize.curve_fit(fitfunction,
                                                  frequencies, 
                                                  np.log(estim_psd_xs_wn/estim_psd_xi_wn),
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))[0]
        ls[i] = popt[0]

        
    return ls



########################################################

ts,upper,lower,unstable = get_equilibria_paths()

true_ls = 1 - upper**2




#numerical simulation of SDEs:

n_steps_per_unit_time = 10               
n_steps = T*n_steps_per_unit_time + 1                                        
solve_ts = np.linspace(0,T,n_steps)
dt = 1/n_steps_per_unit_time


#white noise case

xs_white = np.zeros(n_steps)
xs_white[0] = upper[0]

sigma = 0.01
white_noise = np.random.normal(0,np.sqrt(dt),n_steps)


for i in tqdm.trange(n_steps - 1):
    xs_white[i+1] = xs_white[i] + 0.2*f(xs_white[i],solve_ts[i])*dt + sigma*white_noise[i]                                  



#red noise case

xs_red = np.zeros(n_steps)
red_noise = np.zeros(n_steps)
xs_red[0] = upper[0]                                                           
kappa = 0.05                   


for i in tqdm.trange(n_steps - 1):
    red_noise[i+1] = np.exp(-dt)*red_noise[i] + np.sqrt(1/2*(1-np.exp(-2*dt)))*np.random.normal(0,1)


for i in tqdm.trange(n_steps - 1):
    xs_red[i+1] = xs_red[i] + 0.2*f(xs_red[i],solve_ts[i])*dt + kappa*red_noise[i]*dt


xs_white_filtered = xs_white[::n_steps_per_unit_time]
xs_red_filtered = xs_red[::n_steps_per_unit_time]


white_noise_filtered = np.array([np.sum([white_noise[i*n_steps_per_unit_time+j] for j in range(n_steps_per_unit_time)]) for i in range(T)])
red_noise_filtered = np.array([np.sum([red_noise[i*n_steps_per_unit_time+j] for j in range(n_steps_per_unit_time)])*dt for i in range(T)])




variance_series_white = get_var(xs_white_filtered)
variance_series_red = get_var(xs_red_filtered)
ac_series_white = get_ar(xs_white_filtered)
ac_series_red = get_ar(xs_red_filtered)


ls_white = get_ls(xs_white_filtered,white_noise_filtered,sigma)
ls_red = get_ls(xs_red_filtered,red_noise_filtered,kappa)                         



#showcase of fitting observed psd ration against theoretical one on a windowlength of 1000 with average restoring rate as an approximation to the stationary assumption

xs_white_filt_detr = statsmodels.tsa.tsatools.detrend(xs_white_filtered[:1000],order = 2)

mean_ls_1000 = true_ls[:1000].mean()*0.2
frequencies = 2*np.pi*(1/1000)*np.arange(1,500)

target_psd_ratio = 1/(frequencies**2 + mean_ls_1000**2)

estim_psd_xs_wn = np.array([np.abs(1/np.sqrt(1000)*(np.exp(-1j*frequencies[i]*np.arange(0,1000)) @ xs_white_filt_detr))**2 for i in range(499)])
estim_psd_xi_wn = np.array([np.abs(1/np.sqrt(1000)*(np.exp(-1j*frequencies[i]*np.arange(0,1000)) @ (white_noise_filtered[:1000]*sigma)))**2 for i in range(499)])



#plot observed PSD ratio vs. continuous PSD ratio


plt.plot(frequencies,(target_psd_ratio), color = "blue",label='continuous PSD ratio')
plt.plot(frequencies,(np.array(estim_psd_xs_wn)/np.array(estim_psd_xi_wn)), color = "green",label='observed PSD ratio')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequencies')
plt.legend()

plt.show()


#plot reimplementation of Clarke fig 1


tipp_upper = np.argmin(upper>1)
tipp_lower = np.argmin(lower>-1)


tip_white = np.argmin(xs_white_filtered>1.0)
tip_red = np.argmin(xs_red_filtered>1.0)  


fig,axs = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(8,10)) 

axs[0].plot(ts[:tipp_upper],upper[:tipp_upper],color = "black")
axs[0].plot(ts[tipp_lower:],lower[tipp_lower:],color = "black")
axs[0].plot(ts[~np.isnan(unstable)],unstable[~np.isnan(unstable)],linestyle = "--")
axs[0].plot(ts,xs_red_filtered,color="red")
axs[0].plot(ts,xs_white_filtered,color="blue")


axs[1].plot(ts[window_length:tip_white:window_length],ac_series_white[:int(tip_white/window_length)],color="blue")                               
ax1_var = axs[1].twinx()
ax1_var.plot(ts[window_length:tip_white:window_length],variance_series_white[:int(tip_white/window_length)],color="blue",linestyle="--")


axs[2].plot(ts[window_length:tip_red:window_length],ac_series_red[:int(tip_red/window_length)],color="red")
ax2_var = axs[2].twinx()
ax2_var.plot(ts[window_length:tip_red:window_length],variance_series_red[:int(tip_red/window_length)],color="red",linestyle="--")


axs[0].set_ylabel("x")
axs[1].set_ylabel(r"$AC$")
axs[2].set_ylabel(r"$AC$")
ax1_var.set_ylabel(r"Variance")
ax2_var.set_ylabel(r"Variance")



axs[3].set_xlabel("t")    
axs[3].plot(ts[window_length:tip_white:window_length],ls_white[:int(tip_white/window_length)],color="blue")
axs[3].plot(ts[window_length:tip_red:window_length],ls_red[:int(tip_red/window_length)],color="red")
axs[3].plot(ts[:tipp_upper],-0.2*true_ls[:tipp_upper],color="black")
axs[3].set_ylabel(r"$\lambda$")


ax_0 = axs[0].twiny()
ax_0.set_xlim(-1,1)
ax_0.set_xlabel(r"$\mu(t)$")


plt.show()








