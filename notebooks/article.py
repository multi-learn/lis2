import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import astropy.io.fits as fits

from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    try:
        window_size = np.abs(np.int64(window_size))
        order = np.abs(np.int64(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

"""positions = np.linspace(0, 114000, 114000)
pe = np.absolute((positions - 57000) / 57000)
fig = px.line(x=((57000 - positions) * 0.00319444444400 + 180), 
              y=pe, 
              title="Position encoding",
              width=800,
              height=400,
              labels={
                     "x": "Longitude in degrees",
                     "y": "Posiiton encoding",
                 },)
fig.show()"""

groundtruth_file = "/home/loris/PhD/Dev/Results/nh2_dataset/PE_UNet/PE_UNet_eb_lin/PE_UNet_eb_lin_binarize_segmentation_global_threshold.fits"
groundtruth_image = fits.getdata(groundtruth_file)
idx = np.isnan(groundtruth_image)
groundtruth_image[idx] = 0
groundtruth_image = groundtruth_image[:, 3000 : -3000].copy()
n = groundtruth_image.shape[1]
band_width = 1000
nb_of_bands = int(n / band_width)
lat = np.linspace((57000 - 111000) * 0.00319444444400 + 180, (57000 - 3000) * 0.00319444444400 + 180, nb_of_bands)
tmp = groundtruth_image.copy()
stats = np.array([tmp[:, i * band_width : (i + 1) * band_width].sum() for i in range(nb_of_bands)])
smooth_stats = savitzky_golay(stats, 21, 1)

"""groundtruth_file = "/home/loris/PhD/Dev/Datasets/nh2_dataset/merged/spine_merged.fits"
groundtruth_image = fits.getdata(groundtruth_file)
idx = np.isnan(groundtruth_image)
groundtruth_image[idx] = 0
n = groundtruth_image.shape[0]
band_width = 50
nb_of_bands = int(n / band_width)
lat = np.linspace((-875) * 0.00319444444400 , (875) * 0.00319444444400 , nb_of_bands)
tmp = groundtruth_image.copy()
stats = np.array([tmp[i * band_width : (i + 1) * band_width, :].sum() for i in range(nb_of_bands)])
smooth_stats = savitzky_golay(stats, 11, 4)"""

fig = go.Figure([
    go.Bar(x=lat, y=stats, name="Number of pixel labeled as filament"),
    go.Line(x=lat, y=smooth_stats, name="Savitzky-Golay filter")
])
fig.update_layout(
    title="Number of pixel labeled as filament along galactic longitude",
    xaxis_title="Longitude in degrees",
    yaxis_title="Number of pixel labeled as filament",
    width=900,
    height=400,
)
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="center",
    x=0.5
))
fig.show()

"""plt.plot(lat, stats)
plt.plot(lat, smooth_stats)
plt.vlines(180, ymin=stats.min(), ymax=stats.max(), color="r")
plt.title("Number of pixel labeled as filament along galactic longitude")
plt.show()"""