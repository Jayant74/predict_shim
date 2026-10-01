import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import re
import scipy.io 
os.environ["QT_QPA_PLATFORM"] = "xcb"


class dataloader():
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)
        
        ROOT_PATH = "datasets/"
        OCM_DATA = "OCMDATA/"
        B0DATA = "B0DATA/"

        OCM_PATH = ROOT_PATH + OCM_DATA
        B0_PATH = ROOT_PATH + B0DATA

        ocm_list = []
        b0_list = []

        # Build dictionaries indexed by scan number
        ocm_dict = {}
        for filename in os.listdir(OCM_PATH):
            file_path = os.path.join(OCM_PATH, filename)

            if os.path.isfile(file_path):
                scan_num = self.get_scan_num(filename)

                if scan_num is not None:
                    ocm_dict[scan_num] = file_path


        b0_dict = {}
        for filename in os.listdir(B0_PATH):
            file_path = os.path.join(B0_PATH, filename)

            if os.path.isfile(file_path):
                scan_num = self.get_scan_num(filename)

                if scan_num is not None:
                    b0_dict[scan_num] = file_path


        # Only keep scans that exist in BOTH datasets
        common_scans = sorted(set(ocm_dict) & set(b0_dict))

        ocm_list = [ocm_dict[n] for n in common_scans]
        b0_list  = [b0_dict[n] for n in common_scans]

        # Match OCM and B0 files
        self.io_match = np.array(list(zip(ocm_list, b0_list)))


    # Extract scan number from either naming convention
    def get_scan_num(self, filename):
        match = re.search(r"scan_?(\d+)", filename, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def generate_datasets(self):

        self.all_ocm_windows = []
        self.all_b0_data = []

        for data_idx, dataset in enumerate(self.io_match):

            #print(f"Processing {dataset}") 
            
            # load datasets from matlab file
            ocm = scipy.io.loadmat(dataset[0])
            b0 = scipy.io.loadmat(dataset[1])

            # Assign signals and timestamps
            self.ch1_signal = ocm.get("win_ch1")
            self.ch2_signal = ocm.get("win_ch2")
            self.ch2_tstamps = np.squeeze(ocm.get("win_time"))
            self.bstamps = np.squeeze(ocm.get("dyn_times"))


            # ocm_f = 100Hz, std~ 1e-14 (0.01s)
            # b0_f = 1.31Hz, std~ 1e-15 (0.763 s)

            # Find the period and frequency of the signals
            self.ocm_period = np.mean(self.ch2_tstamps[1:] - self.ch2_tstamps[:-1]) # sec
            self.ocm_freq = 1/self.ocm_period # Hz

            self.b0_period = np.mean(self.bstamps[1:] - self.bstamps[:-1]) # sec
            self.b0_freq = 1/self.b0_period    # Hz   

            print(f"OCM frequency = {self.ocm_freq} Hz")
            print(f"b0 frequency = {self.b0_freq} Hz")
            print(f"OCM frequency std = {np.std((self.ch2_tstamps[1:] - self.ch2_tstamps[:-1]))}")
            print(f"B0frequency std = {np.std((self.bstamps[1:] - self.bstamps[:-1]))}")
            print(f"OCM Period (s) = {self.ocm_period}")
            print(f"B0 Period (s) = {self.b0_period}")

            # Fill in arrays for the Timeindices of B0 signal in OCM data
            ocm_b0_tstamp = np.zeros((self.bstamps.shape))
            ocm_b0_tindex = np.zeros(self.bstamps.shape, dtype=int)

            for i, btime in enumerate(self.bstamps):
                ky_n = self.ch2_tstamps - btime

                ocm_ky_n_index = np.argwhere((ky_n <= 1e-2) & (ky_n > 0))
                idx = ocm_ky_n_index[0, 0]
                ocm_ky_n_tstamp = self.ch2_tstamps[idx]

                ocm_b0_tindex[i] = idx
                ocm_b0_tstamp[i] = self.ch2_tstamps[idx]

            bmap = b0.get('masked_B0_data')

            # ADDED: transpose B0 so dimension 0 corresponds to the B0/window number
            b0_data = bmap.T

            # Window maker
            num_windows = self.bstamps.shape[0]
            ocm_dim = self.ch2_signal.shape[0]

            win_period = self.b0_period - self.ocm_period
            win_len = int(win_period / self.ocm_period)

            N = self.bstamps.shape[0]

            # Storage for this dataset only
            # We are filling in the windows of OCM data for every B0 map for a scan 
            # [Number of B0 maps, OCM dimension (depth), Window Length, Complex]
            ocm_window_data = np.zeros(
                (N, ocm_dim, win_len),
                dtype=self.ch2_signal.dtype
            )

            # CHANGED: use range(N) so the final allocated window is also filled
            for i in range(N):

                # Start at the first OCM shot right after the B0 map was taken
                end = int(ocm_b0_tindex[i] + 1)
                start = end - win_len

                # If the first B0 timestamp was taken before a full window-length of OCM data was recorded
                # then make a window sized array and pad the available OCM data.
                if start < 0:
                    print("Padding the first OCM data")
                    # Available samples before B0
                    available = self.ch2_signal[:, 0:end]

                    # Create zero-padded window
                    ocm_window = np.zeros(
                        (ocm_dim, win_len),
                        dtype=self.ch2_signal.dtype
                    )

                    # Put available samples at the end
                    ocm_window[:, -available.shape[1]:] = available

                # Take the OCM data. 
                else:
                    #end = int(ocm_b0_tindex[i]+1)
                    #start = int(end - win_len)
                    ocm_window = self.ch2_signal[:, start:end]

                # This fills the [Num B0 maps, depth, window length]
                ocm_window_data[i] = ocm_window

            # Existing OCM storage
            self.all_ocm_windows.append(ocm_window_data)
            # ADDED: save the corresponding B0 dataset instead of losing it next iteration
            self.all_b0_data.append(b0_data)


    def training_data(self, dataset_num):
        
        self.training_data_ocm = torch.as_tensor(
            np.abs(self.all_ocm_windows[dataset_num]),
            dtype=torch.float64,
            device=self.device
        )

        self.training_data_b0 = torch.as_tensor(
            self.all_b0_data[dataset_num],
            dtype=torch.float64,
            device=self.device
        )
        
    def test_data(self, dataset_num):
        
        self.test_data_ocm = torch.as_tensor(
            np.abs(self.all_ocm_windows[dataset_num]),
            dtype=torch.float64,
            device=self.device
        )



        self.test_data_b0 = torch.as_tensor(
            self.all_b0_data[dataset_num],
            dtype=torch.float64,
            device=self.device
        )

    def flatten_data(self):

        # One flattened OCM signal window for each B0 signal
        num_b0, ocm_dim, win_len = self.training_data_ocm.shape

        self.Ut = self.training_data_ocm.reshape(
            num_b0,
            ocm_dim * win_len
        )

        self.UT = self.test_data_ocm.reshape(
            self.test_data_ocm.shape[0],
            ocm_dim * win_len
        )

    def noise_scan(self, dataset_num):

        # Number of samples in 5 seconds
        noise_samples = int(5 / self.ocm_period)

        # First 5 seconds of raw OCM acquisition
        self.noise_data = torch.as_tensor(
            np.abs(self.ch2_signal[:, :noise_samples]),
            dtype=torch.float64,
            device=self.device
        )


    # CHANGED: moved here because ocm_window now exists
    #print(f"# OCM signals * ocm_period = {ocm_window.shape[1] * ocm_period} seconds")

    def visualize(self, bmap, ch1_signal, ch2_signal):

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



    '''plt.figure()
    plt.vlines(ch1_tstamps,ymin=0,ymax=5)
    plt.vlines(ocm_b0_tstamp,ymin=0,ymax=10)
    plt.show()'''

# -------------------------
# Device
# -------------------------




# Expected: (197, 500)

'''
OCM frequency = 100.0 Hz
b0 frequency = 1.4598540145985388 Hz
OCM frequency std = 4.4555958862654946e-14
B0frequency std = 9.746538369909138e-14
OCM Period (s) = 0.01
B0 Period (s) = 0.6850000000000006
'''


# -------------------------
# Covariance
# -------------------------


