import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import re
import scipy.io 
os.environ["QT_QPA_PLATFORM"] = "xcb"

ROOT_PATH = "datasets/"
OCM_DATA = "OCMDATA/"
B0DATA = "B0DATA/"

OCM_PATH = ROOT_PATH + OCM_DATA
B0_PATH = ROOT_PATH + B0DATA

def visualize(bmap, ch1_signal, ch2_signal):

    ch1_signal = np.abs(ch1_signal)
    ch2_signal = np.abs(ch2_signal)


    plt.figure()
    plt.imshow(bmap, aspect="auto")
    plt.title("B0 Map")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")


    plt.figure()
    plt.imshow(ch1_signal, aspect="auto")
    plt.title("OCM Abdomen Map")
    plt.xlabel("time (ms)")
    plt.ylabel("y (depth)")

    plt.figure()
    plt.imshow(ch2_signal, aspect="auto")
    plt.title("OCM Chest Map")
    plt.xlabel("time (ms)")
    plt.ylabel("y (depth)")

    # 1. Create the figure and a 3D axis
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection='3d')


    rows, cols = ch1_signal.shape
    x = np.arange(0, cols)  # time 
    y = np.arange(0, rows)  # depth
    X, Y = np.meshgrid(x, y)
    N = 1

    depths = [5, 10, 20, 50, 80, 110, 150, 160]
    # 4. Plot the 3D surface
    ch1_centered = ch1_signal - np.mean(ch1_signal[:, :, np.newaxis],axis=1)
    mask= np.zeros((ch1_centered.shape))
    mask[[5, 10, 20, 50, 80, 110, 150, 160], :] = np.ones((1,ch1_centered.shape[1]))
    ch1_masked = ch1_centered * mask
    X =X[:, ::N]
    Y = Y[:, ::N]
    surf = ax.plot_surface(X, Y, ch1_masked[:, ::N], cmap='plasma', edgecolor='none',alpha=0.5,linewidth=1)

    # 5. Add a color bar and labels
    ax.set_title('3D Surface Plot')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('delta_x')
    ax.set_zlabel('Intensity')

    # 2. Crop out the default oversized 3D margins 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # 3. Zoom/Scale the axes relative to the view bounding box
    # (Values < 1 zoom in, default is 1, 1, 1)
    ax.set_box_aspect(None, zoom=0.85) 

    plt.show()

ocm_list = []
b0_list = []

# Extract scan number from either naming convention
def get_scan_num(filename):
    match = re.search(r"scan_?(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


# Build dictionaries indexed by scan number
ocm_dict = {}
for filename in os.listdir(OCM_PATH):
    file_path = os.path.join(OCM_PATH, filename)

    if os.path.isfile(file_path):
        scan_num = get_scan_num(filename)

        if scan_num is not None:
            ocm_dict[scan_num] = file_path


b0_dict = {}
for filename in os.listdir(B0_PATH):
    file_path = os.path.join(B0_PATH, filename)

    if os.path.isfile(file_path):
        scan_num = get_scan_num(filename)

        if scan_num is not None:
            b0_dict[scan_num] = file_path


# Only keep scans that exist in BOTH datasets
common_scans = sorted(set(ocm_dict) & set(b0_dict))

ocm_list = [ocm_dict[n] for n in common_scans]
b0_list  = [b0_dict[n] for n in common_scans]

# Match OCM and B0 files
io_match = np.array(list(zip(ocm_list, b0_list)))

all_ocm_windows = []
all_b0_data = []

for data_idx, dataset in enumerate(io_match):
    #print(f"Processing {dataset}") 
    
    # load datasets from matlab file
    ocm = scipy.io.loadmat(dataset[0])
    b0 = scipy.io.loadmat(dataset[1])

    # Assign signals and timestamps
    ch1_signal = ocm.get("win_ch1")
    ch2_signal = ocm.get("win_ch2")
    ch2_tstamps = np.squeeze(ocm.get("win_time"))
    bstamps = np.squeeze(ocm.get("dyn_times"))


    # ocm_f = 100Hz, std~ 1e-14 (0.01s)
    # b0_f = 1.31Hz, std~ 1e-15 (0.763 s)

    # Ignore these for now, I am only using the timestamp of when the image was formed (k=+n)
    TE = 0.001 # Period of 1 TE
    TR = 0.01  # Period of 1 TR
    N_TR = 100 # Number of TRs
    Period_image = TR * N_TR


    # Find the period and frequency of the signals
    ocm_period = np.mean(ch2_tstamps[1:] - ch2_tstamps[:-1]) # sec
    ocm_freq = 1/ocm_period # Hz

    b0_period = np.mean(bstamps[1:] - bstamps[:-1]) # sec
    b0_freq = 1/b0_period    # Hz   

    print(f"OCM frequency = {ocm_freq} Hz")
    print(f"b0 frequency = {b0_freq} Hz")
    print(f"OCM frequency std = {np.std((ch2_tstamps[1:] - ch2_tstamps[:-1]))}")
    print(f"B0frequency std = {np.std((bstamps[1:] - bstamps[:-1]))}")
    print(f"OCM Period (s) = {ocm_period}")
    print(f"B0 Period (s) = {b0_period}")

    # Ignore
    ocm_win_time_width = TE
    ocm_win_half_time_width = TE / 2
    ocm_win_index_width = TE * ocm_freq



    # Fill in arrays for the Timeindices of B0 signal in OCM data
    ocm_b0_tstamp = np.zeros((bstamps.shape))
    ocm_b0_tindex = np.zeros((bstamps.shape))

    for i, btime in enumerate(bstamps):
        ky_n = ch2_tstamps - bstamps[i]
        ky_0 = ch2_tstamps - (bstamps[i] - TE/2)

        ocm_ky_n_index = np.argwhere((ky_n <= 1e-2) & (ky_n > 0))
        idx = ocm_ky_n_index[0, 0]
        ocm_ky_n_tstamp = ch2_tstamps[idx]

        ocm_b0_tindex[i] = idx
        ocm_b0_tstamp[i] = ch2_tstamps[idx]

    bmap = b0.get('masked_B0_data')
    bmap_signal_num = bmap.shape[1]
    bmap_signal = bmap.shape[0]

    # ADDED: transpose B0 so dimension 0 corresponds to the B0/window number
    b0_data = bmap.T


    # CODE ABSOLUTELY WORKS UP UNTIL HERE
    #visualize(bmap=bmap, ch1_signal=ch1_signal, ch2_signal=ch2_signal)
    #plt.show()


    # Window maker
    num_windows = bstamps.shape[0]
    ocm_dim = ch2_signal.shape[0]

    win_period = b0_period - ocm_period
    win_len = int(win_period / ocm_period)

    N = bstamps.shape[0]

    # Storage for this dataset only
    # We are filling in the windows of OCM data for every B0 map for a scan 
    # [Number of B0 maps, OCM dimension (depth), Window Length, Complex]
    ocm_window_data = np.zeros(
        (N, ocm_dim, win_len),
        dtype=ch2_signal.dtype
    )

    #visualize(bmap=bmap, ch1_signal=ch1_signal, ch2_signal=ch2_signal)

    # CHANGED: use range(N) so the final allocated window is also filled
    for i in range(N):

        # Start at the first OCM shot right after the B0 map was taken
        end = int(ocm_b0_tindex[i] + 1)
        start = end - win_len

        # If the first B0 timestamp was taken before a full window-length of OCM data was recorded
        # then make a window sized array and pad the available OCM data.
        if start < 0:
            print("FLAG FLAG FLAG")
            # Available samples before B0
            available = ch2_signal[:, 0:end]

            # Create zero-padded window
            ocm_window = np.zeros(
                (ocm_dim, win_len),
                dtype=ch2_signal.dtype
            )

            # Put available samples at the end
            ocm_window[:, -available.shape[1]:] = available

        # Take the OCM data. 
        else:
            #end = int(ocm_b0_tindex[i]+1)
            #start = int(end - win_len)
            ocm_window = ch2_signal[:, start:end]

        # This fills the [Num B0 maps, depth, window length]
        ocm_window_data[i] = ocm_window

        # Existing OCM storage
    all_ocm_windows.append(ocm_window_data)

    # ADDED: save the corresponding B0 dataset instead of losing it next iteration
    all_b0_data.append(b0_data)

    print(f"{dataset} OCM window data shape: {ocm_window_data.shape}")

    # CHANGED: moved here because ocm_window now exists
    #print(f"# OCM signals * ocm_period = {ocm_window.shape[1] * ocm_period} seconds")



    '''plt.figure()
    plt.vlines(ch1_tstamps,ymin=0,ymax=5)
    plt.vlines(ocm_b0_tstamp,ymin=0,ymax=10)
    plt.show()'''

import numpy as np
import torch
import matplotlib.pyplot as plt



# -------------------------
# Device
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -------------------------
# Load datasets as tensors
# -------------------------

training_data_ocm = torch.as_tensor(
    np.abs(all_ocm_windows[1]),
    dtype=torch.float64,
    device=device
)

test_data_ocm = torch.as_tensor(
    np.abs(all_ocm_windows[3]),
    dtype=torch.float64,
    device=device
)


training_data_b0 = torch.as_tensor(
    all_b0_data[0],
    dtype=torch.float64,
    device=device
)

test_data_b0 = torch.as_tensor(
    all_b0_data[2],
    dtype=torch.float64,
    device=device
)

noise_scan1 = torch.as_tensor(
    all_ocm_windows[1],
    dtype=torch.float64,
    device=device
)


'''
OCM frequency = 100.0 Hz
b0 frequency = 1.4598540145985388 Hz
OCM frequency std = 4.4555958862654946e-14
B0frequency std = 9.746538369909138e-14
OCM Period (s) = 0.01
B0 Period (s) = 0.6850000000000006
'''
ocm_period = 0.01

# Number of samples in 5 seconds
noise_samples = int(5 / ocm_period)

# First 5 seconds of raw OCM acquisition
noise_data = torch.as_tensor(
    np.abs(ch2_signal[:, :noise_samples]),
    dtype=torch.float64,
    device=device
)

print(noise_data.shape)
# Expected: (197, 500)



# -------------------------
# Covariance
# -------------------------

def covariance(Ut):

    # Center each feature
    Ut_centered = Ut - torch.mean(Ut, dim=0, keepdim=True)

    # Equivalent to np.cov(Ut, rowvar=False)
    return (
        Ut_centered.T @ Ut_centered
        / (Ut.shape[0] - 1)
    )


# -------------------------
# Flatten OCM windows
# -------------------------

# One flattened OCM signal window for each B0 signal
num_b0, ocm_dim, win_len = training_data_ocm.shape

Ut = training_data_ocm.reshape(
    num_b0,
    ocm_dim * win_len
)

UT = test_data_ocm.reshape(
    test_data_ocm.shape[0],
    ocm_dim * win_len
)


# Standard deviation over the 500 temporal noise samples
sigma_ocm = torch.std(
    noise_data,
    dim=1
)

print(sigma_ocm.shape)
# (197,)


# Repeat each depth's sigma across the 75 samples in the KDE window
sigma_ocm = sigma_ocm[:, None].expand(
    ocm_dim,
    win_len
)

print(sigma_ocm.shape)
# (197, 75)


# Flatten in exactly the same order as Ut
sigma_ocm = sigma_ocm.reshape(-1)

print(sigma_ocm.shape)
# (14775,)

print(Ut.shape)
# (170, 14775)

# Flatten identically to Ut
# Result: (ocm_dim * win_len,)
sigma_ocm = sigma_ocm.reshape(-1)

# Prevent division by zero
sigma_ocm = torch.clamp(
    sigma_ocm,
    min=1e-8
)

print(sigma_ocm.shape)
print(Ut.shape)


# Ut is prior sample, UT is new sample
def eq5(Ut, UT, sigma_ocm):

    diff = Ut - UT

    # Equivalent to:
    # diff.T @ Sigma_inv @ diff
    # when Sigma = diag(sigma_ocm**2)
    mahalanobis = torch.sum(
        (diff / sigma_ocm) ** 2
    )

    return torch.exp(
        -0.5 * mahalanobis
    )


def eq6(I, U, UT, sigma_ocm):

    # Numerator has same shape as one B0
    Pj = torch.zeros_like(
        I[0],
        dtype=torch.float64,
        device=device
    )

    for t, Ut in enumerate(U):

        weight = eq5(
            Ut,
            UT,
            sigma_ocm
        )

        Pj += weight * I[t]

    return Pj


def eq7(U, UT, sigma_ocm):

    Pc = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device
    )

    for Ut in U:

        Pc += eq5(
            Ut,
            UT,
            sigma_ocm
        )

    return Pc


def E_i(I, U, UT, sigma_ocm, h):
    '''
    I - 170 training images
    U - 170 training OCM windows
    UT - new OCM window
    '''
    diff = U - UT

    d2 = torch.sum(
        (diff / sigma_ocm) ** 2,
        dim=1
    )

    weights = torch.softmax(
        -0.5 * d2 / (h ** 2),
        dim=0
    )

    weights = weights.unsqueeze(1)

    return torch.sum(weights * I, dim=0)
# Run KDE on test samples
# -------------------------

predicted_b0 = torch.zeros_like(
    test_data_b0,
    dtype=torch.float64,
    device=device
)

for i, UT_i in enumerate(UT):

    predicted_b0[i] = E_i(
        training_data_b0,
        Ut,
        UT_i,
        sigma_ocm,
        h=513
    )



print("Predicted B0:", predicted_b0.shape)


# -------------------------
# MSE per B0
# -------------------------

# Average over every dimension except B0/sample dimension
mse_dims = tuple(
    range(1, predicted_b0.ndim)
)

mse_per_b0 = torch.mean(
    (predicted_b0 - test_data_b0) ** 2,
    dim=mse_dims
)

print("MSE per B0:", mse_per_b0)


# -------------------------
# Move results to CPU if needed
# -------------------------

predicted_b0_np = (
    predicted_b0
    .detach()
    .cpu()
    .numpy()
)

mse_per_b0_np = (
    mse_per_b0
    .detach()
    .cpu()
    .numpy()
)


# Mean training B0
mean_b0 = training_data_b0.mean(dim=0)

# Baseline prediction for every test B0
baseline_b0 = mean_b0.unsqueeze(0).expand_as(test_data_b0)

# Baseline MSE
baseline_mse = torch.mean(
    (baseline_b0 - test_data_b0) ** 2,
    dim=tuple(range(1, test_data_b0.ndim))
)

kde_mse = mse_per_b0.mean()
baseline_mse_mean = baseline_mse.mean()

kde_rmse = torch.sqrt(kde_mse)
baseline_rmse = torch.sqrt(baseline_mse_mean)

UT_i = UT[0]

weights = []

for Ut_i in Ut:

    diff = Ut_i - UT_i

    d2 = torch.sum(
        (diff / sigma_ocm) ** 2
    )

    weight = torch.exp(
        -0.5 * d2
    )

    weights.append(weight)

weights = torch.stack(weights)

# Normalize weights
alpha = weights / torch.sum(weights)

print("Sum:", alpha.sum().item())

values, indices = torch.topk(alpha, 10)

print("Top weights:", values)
print("Training B0 indices:", indices)


print("KDE MSE:", kde_mse.item())
print("Baseline MSE:", baseline_mse_mean.item())

print("KDE RMSE:", kde_rmse.item())
print("Baseline RMSE:", baseline_rmse.item())

print(
    "MSE ratio:",
    (kde_mse / baseline_mse_mean).item()
)

improvement = (
    1 -
    mse_per_b0.mean() / baseline_mse.mean()
) * 100

print("MSE improvement over mean baseline:", improvement.item(), "%")


rmse_per_b0 = torch.sqrt(mse_per_b0)

print("MSE median:", mse_per_b0.median().item())
print("MSE mean:", mse_per_b0.mean().item())

print("RMSE median:", rmse_per_b0.median().item())
print("RMSE mean:", rmse_per_b0.mean().item())

print("RMSE min:", rmse_per_b0.min().item())
print("RMSE max:", rmse_per_b0.max().item())

print(
    "Percentage RMSE < 8:",
    (rmse_per_b0 < 8).double().mean().item() * 100
)

print(
    "Percetange RMSE < 5:",
    (rmse_per_b0 < 5).double().mean().item() * 100
)

plt.figure()
plt.plot(np.arange(mse_per_b0.detach().cpu().numpy().shape[0]), rmse_per_b0.detach().cpu().numpy())
plt.xlabel("B0 Index")
plt.ylabel("RMSE")
plt.show()

# Toy problem: define distributions for toy relationships

# CNN KDE


'''

# Ut is prior sample, UT is new sample
def eq5(Ut, UT, sigma_ocm):

    diff = Ut - UT

    # Equivalent to:
    # diff.T @ Sigma_inv @ diff
    # when Sigma = diag(sigma_ocm**2)
    mahalanobis = torch.sum(
        (diff / sigma_ocm) ** 2
    )

    return torch.exp(
        -0.5 * mahalanobis
    )


def eq6(I, U, UT, sigma_ocm):

    # Numerator has same shape as one B0
    Pj = torch.zeros_like(
        I[0],
        dtype=torch.float64,
        device=device
    )

    for t, Ut in enumerate(U):

        weight = eq5(
            Ut,
            UT,
            sigma_ocm
        )

        Pj += weight * I[t]

    return Pj


def eq7(U, UT, sigma_ocm):

    Pc = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device
    )

    for Ut in U:

        Pc += eq5(
            Ut,
            UT,
            sigma_ocm
        )

    return Pc


def E_i(I, U, UT, sigma_ocm):

    return (
        eq6(I, U, UT, sigma_ocm)
        /
        eq7(U, UT, sigma_ocm)
    )


# -------------------------
# Covariance + pseudoinverse
# -------------------------

Ut_cov = covariance(Ut)

Ut_cov_inv = torch.linalg.pinv(Ut_cov)

print("Covariance:", Ut_cov.shape)


# -------------------------
# Plot covariance
# -------------------------

# Move only the portion needed for plotting back to CPU
Ut_cov_plot = (
    Ut_cov[0:4000, 0:4000]
    .detach()
    .cpu()
    .numpy()
)

plt.figure()
plt.imshow(Ut_cov_plot, aspect="auto")
plt.title("Covariance")
plt.xlabel("ocm feature")
plt.ylabel("ocm feature")
plt.show()


# -------------------------
# Gaussian normalization
# -------------------------

def compute_v(Ut):

    # Number of dimensions/features
    n = Ut.shape[0]

    # Determinant of covariance matrix
    det = torch.linalg.det(Ut)

    # Gaussian normalization denominator
    v = torch.sqrt(
        (2 * torch.pi) ** n * det
    )

    return v


# -------------------------
# Equation 5
# -------------------------

# Ut is prior sample, UT is new sample
def eq5(Ut, UT, Ut_cov_in):

    diff = Ut - UT

    # Mahalanobis quadratic form
    return torch.exp(
        -0.5 * (diff.T @ Ut_cov_in @ diff)
    )


# -------------------------
# Equation 6
# -------------------------

def eq6(I, U, UT, Sigma_inv):

    # Numerator has the same shape as one B0 image
    Pj = torch.zeros_like(
        I[0],
        dtype=I.dtype,
        device=I.device
    )

    for t, Ut in enumerate(U):

        # Kernel weight for this U-I pair
        weight = eq5(
            Ut,
            UT,
            Sigma_inv
        )

        # Weight matching B0 image
        Pj += weight * I[t]

    return Pj


# -------------------------
# Equation 7
# -------------------------

def eq7(U, UT, Sigma_inv):

    # Scalar denominator on GPU
    Pc = torch.zeros(
        (),
        dtype=U.dtype,
        device=U.device
    )

    for Ut in U:

        Pc += eq5(
            Ut,
            UT,
            Sigma_inv
        )

    return Pc


# -------------------------
# Conditional expectation
# -------------------------

def E_i(I, U, UT, Sigma_inv):

    return (
        eq6(I, U, UT, Sigma_inv)
        /
        eq7(U, UT, Sigma_inv)
    )

    
predicted_b0 = torch.zeros_like(
    test_data_b0,
    dtype=torch.float64,
    device=device
)

for i, UT_i in enumerate(UT):

    predicted_b0[i] = E_i(
        training_data_b0,
        Ut,
        UT_i,
        Ut_cov_inv
    )
'''

# DATA STRUCTURE
'''

Okay so how should this data be structured?

Bruno's paper did:

OCM signal from Ky=0 of Image 1 to Ky=0 of Image 2
MRI Image from Ky=-n to Ky=+n of Image 2

This data temporal mixing of the data pairs is structured specifically 
for predictive power. At Ky=0 most of the MR image is acquried, so just before that TR
the prior ultrasound signal is associated with that image. 



'''

'''
170 images, taken 0.763 seconds apart, Un, In = I_tstamp1 - 
'''


'''
readout_duration = 4e-1 # s
TE = 4e-1

U_freq = (ch1_tstamps[1:] - ch1_tstamps[:-1]) / ch1_tstamps.shape[0] # Hz
b0_freq = (bstamps[1:] - bstamps[:-1]) / bstamps.shape[0] # Hz

bstamps_ky0 = bstamps - (readout_duration / 2)
bstamps_ky0 = bstamps - (TE / 2)

bstamps_ky0_time_width = (bstamps_ky0[1:] - bstamps_ky0[:-1]) / bstamps_ky0.shape[0]
ocm_ky0_index_width = int(bstamps_ky0_time_width / ocm_period)

for i, btime in enumerate(bstamps):
    ocm_ky_0_index = np.argwhere((bstamps_ky0 <= 5e-3) & (bstamps_ky0 > 0))
    ocm_ky_0_tstamp = ch1_tstamps[ocm_ky_n_index]
    ocm_tindex[i] = ocm_ky_n_index[0,0] 

TR = 100




# timestamp@ky=-n - 1ms = ky=0 -> Get ky=0 for every image -> {I2, U[ky=0@I1:ky=0@I2]} = data pairs
# U_ky [:, 1:2],    [:, 2:3],   [:, 3:4] ...     [n:n+1]
# I      2              3           4    ...     [n+1]

# VANILLA KDE
# v = np.sqrt(((2*np.pi)**n ) * (np.linalg.eigh(Sigma)) <- what dimension is this?  
# N(Ut; UT, Sigma) = v * np.exp(-0.5 * (Ut - UT).T * Sigma_inv * (Ut - UT))
# Pj(Pj; Nt, It, N(Ut; UT, Sigma)) => jd = np.zeros(It.size); for i in range(Nt): jd += (1/Nt) * (It * N(Ut; UT, Sigma))
# Pc(Pc; Nt, N(Ut; UT, Sigma)) -> pj = np.zeros(Ut.size); for i in range(Nt): pj += (1/*Nt) * N(Ut; UT, Sigma)
# E[It; Ut, D] = Pj * 1/Pc -> It.shape


# Mock data
rng = np.random.default_rng(seed=42)  
U = rng.standard_normal((197, 13000))
I = rng.standard_normal(48, 170)

bstamp_ky = np.arange(0,180,step=1.058) # MRI simulated ky=0 timestamps
ocm_ky0 = np.arange(0,180,step=1.058) # OCM simulated ky=0 timestamps 

# Build sliding window function here
win_num = 5 # Say index width of 5 equals the number of OCM timestamps from ky_t to ky_t+1
for t in range(bstamp_ky.shape[0]):
    U[t] = ch1_signal[:, t * win_num:win_num*(t+1)]  # OCM data from t to t+1

# U = [170 - 1, 193, int(ky_period/OCM_period)] Dim[# of images - 1, OCM depth dim, # ocm shots per ky width]
# I = [170 - 1, 50] Dim[# of images - 1, length of pixels]

# U.shape = [N_t - 1, Ud, N_TR] -> concatenate -> [N_t, Ud * N_TR]
# I.shape = [N_t - 1, Id]

Sigma = np.cov(U[:, :])  # Sigma.shape = [N_t - 1, (Ud * N_TR)]
Sigma_mag = np.linalg.det(Sigma[:, :]) # Sigma_mag.shape = [N_t - 1, 1]

v = 1 / np.sqrt(((2*np.pi)**(Ud/2)) * np.sqrt(Sigma_mag))
Sigma_inv = np.linalg.pinv(Sigma)

#N(Ut; UT, Sigma) 

'''


"""
ocm_list = []
b0_list = []

# Extract scan number from either naming convention
def get_scan_num(filename):
    match = re.search(r"scan_?(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


# Build dictionaries indexed by scan number
ocm_dict = {}
for filename in os.listdir(OCM_PATH):
    file_path = os.path.join(OCM_PATH, filename)

    if os.path.isfile(file_path):
        scan_num = get_scan_num(filename)

        if scan_num is not None:
            ocm_dict[scan_num] = file_path


b0_dict = {}
for filename in os.listdir(B0_PATH):
    file_path = os.path.join(B0_PATH, filename)

    if os.path.isfile(file_path):
        scan_num = get_scan_num(filename)

        if scan_num is not None:
            b0_dict[scan_num] = file_path


# Only keep scans that exist in BOTH datasets
common_scans = sorted(set(ocm_dict) & set(b0_dict))

ocm_list = [ocm_dict[n] for n in common_scans]
b0_list  = [b0_dict[n] for n in common_scans]

# Match OCM and B0 files
io_match = np.array(list(zip(ocm_list, b0_list)))

all_ocm_windows = []
all_b0_data = []

for data_idx, dataset in enumerate(io_match):
    #print(f"Processing {dataset}") 
    ocm = scipy.io.loadmat(dataset[0])
    b0 = scipy.io.loadmat(dataset[1])

    ch1_signal = ocm.get("win_ch1")
    ch2_signal = ocm.get("win_ch2")
    ch2_tstamps = np.squeeze(ocm.get("win_time"))
    bstamps = np.squeeze(ocm.get("dyn_times"))


    # ocm_f = 100Hz, std~ 1e-14 (0.01s)
    # b0_f = 1.31Hz, std~ 1e-15 (0.763 s)

    TE = 0.001 # Period of 1 TE
    TR = 0.01  # Period of 1 TR
    N_TR = 100 # Number of TRs
    Period_image = TR * N_TR
    
    ocm_period = np.mean(ch2_tstamps[1:] - ch2_tstamps[:-1]) # sec
    ocm_freq = 1/ocm_period # Hz

    b0_period = np.mean(bstamps[1:] - bstamps[:-1]) # sec
    b0_freq = 1/b0_period    # Hz   

    print(f"OCM frequency = {ocm_freq} Hz")
    print(f"b0 frequency = {b0_freq} Hz")
    print(f"OCM frequency std = {np.std((ch2_tstamps[1:] - ch2_tstamps[:-1]))}")
    print(f"B0frequency std = {np.std((bstamps[1:] - bstamps[:-1]))}")
    print(f"OCM Period (s) = {ocm_period}")
    print(f"B0 Period (s) = {b0_period}")


    ocm_win_time_width = TE
    ocm_win_half_time_width = TE / 2

    ocm_win_index_width = TE * ocm_freq

    # NEED THE MRI READOUT TIMESTAMPS

    # Timeindex of B0 signal in OCM data
    ocm_b0_tstamp = np.zeros((bstamps.shape))
    ocm_b0_tindex = np.zeros((bstamps.shape))

    for i, btime in enumerate(bstamps):
        ky_n = ch2_tstamps - bstamps[i]
        ky_0 = ch2_tstamps - (bstamps[i] - TE/2)

        ocm_ky_n_index = np.argwhere((ky_n <= 1e-2) & (ky_n > 0))
        idx = ocm_ky_n_index[0, 0]
        ocm_ky_n_tstamp = ch2_tstamps[idx]

        ocm_b0_tindex[i] = idx
        ocm_b0_tstamp[i] = ch2_tstamps[idx]

    bmap = b0.get('masked_B0_data')
    bmap_signal_num = bmap.shape[1]
    bmap_signal = bmap.shape[0]
    #visualize(bmap=bmap, ch1_signal=ch1_signal, ch2_signal=ch2_signal)
    #plt.show()

    print(f"# OCM signals * ocm_period = {ocm_window.shape[1] * ocm_period} seconds")

    # Window maker
    num_windows = bstamps.shape[0]
    ocm_dim = ch2_signal.shape[0]

    win_period = b0_period - ocm_period
    win_len = int(win_period / ocm_period)

    N = bstamps.shape[0]

    # Storage for this dataset only
    ocm_window_data = np.zeros(
        (N, ocm_dim, win_len),
        dtype=ch2_signal.dtype
    )

    for i in range(N - 1):

        end = int(ocm_b0_tindex[i] + 1)
        start = end - win_len

        if start < 0:
            print("FLAG FLAG FLAG")
            # Available samples before B0
            available = ch2_signal[:, 0:end]

            # Create zero-padded window
            ocm_window = np.zeros(
                (ocm_dim, win_len),
                dtype=ch2_signal.dtype
            )

            # Put available samples at the end
            ocm_window[:, -available.shape[1]:] = available

        else:
            end = int(ocm_b0_tindex[i]+1)
            start = int(end - win_len)
            ocm_window = ch2_signal[:, start:end]


        ocm_window_data[i] = ocm_window

    all_ocm_windows.append(ocm_window_data)


    print(f"{dataset} OCM window data shape: {ocm_window_data.shape}")
    #print(f"B0 data shape: {b0_data.shape}")

    print()
    print()
    '''plt.figure()
    plt.vlines(ch1_tstamps,ymin=0,ymax=5)
    plt.vlines(ocm_b0_tstamp,ymin=0,ymax=10)
    plt.show()'''

for scan in range(ocm_window_data.shape[0]):
    for i in range(ocm_window_data.shape[1]):
        if np.max(np.abs(ocm_window_data[scan, i])) == 0:
            print("ZERO:", scan, i)

# First divide into a Priori, D, and test set 
training_data_ocm = torch.Tensor(ocm_window_data[:2, :, :, :])
training_data_b0 = torch.Tensor(b0_data[:2, :, :])

test_data_ocm = torch.Tensor(ocm_window_data[2:, :, :, :])
test_data_b0 = torch.Tensor(b0_data[2:, :, :])

def covariance(Ut):
    return np.cov(Ut)

def normalization_coefficient(sigma, n):
    v = np.sqrt(((2*np.pi)**n ) * (np.linalg.eigh(sigma)))
    return v

# Flatten so you have 1 flattened OCM signal window for each B0 signal
scan_num, num_b0, ocm_dim, win_len = training_data_ocm.shape

Ut = training_data_ocm.reshape(
    scan_num * num_b0,
    ocm_dim * win_len
)"""