from matplotlib import pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class OCMFeatureCNN(nn.Module):
    def __init__(self, n_channels, latent_dim=32):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = self.cnn(x).squeeze(-1)
        z = self.fc(x)

        return z


class KDE():
    def __init__(self):
        pass

    def kde_predict(self, query_z, reference_z, reference_b0, h=0.5):

        # Distance between learned OCM features
        # [N_query, N_reference]
        d2 = torch.cdist(query_z, reference_z).pow(2)
        '''
        query_z = F.normalize(query_z, p=2, dim=1)
        reference_z = F.normalize(reference_z, p=2, dim=1)

        similarity = query_z @ reference_z.T
        d2 = 1 - similarity
        '''
        # KDE weights
        weights = torch.softmax(
            -0.5 * d2 / h**2,
            dim=1
        )

        # Weighted sum of B0 maps
        b0_shape = reference_b0.shape[1:]
        reference_b0 = reference_b0.reshape(reference_b0.shape[0], -1)

        predicted_b0 = weights @ reference_b0

        return predicted_b0.reshape(
            query_z.shape[0],
            *b0_shape
        )


# --------------------------------------------------
# Setup
# --------------------------------------------------

class train():
    def __init__(
        self,
        training_data_ocm,
        training_data_b0,
        test_data_ocm,
        test_data_b0,
        CNN,
        device
    ):
        self.device = device

        self.training_data_ocm = training_data_ocm
        self.training_data_b0 = training_data_b0

        self.test_data_ocm = test_data_ocm
        self.test_data_b0 = test_data_b0

        self.model = CNN(
            n_channels=training_data_ocm.shape[1],
            latent_dim=32
        ).to(device).double()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    def train_model(self):

        loss_history = []
        rmse_history = []

        for epoch in range(500):

            self.model.train()
            self.optimizer.zero_grad()

            # [N_train, channels, time]
            #           ↓ CNN
            # [N_train, latent_dim]
            z = self.model(self.training_data_ocm)

            # Pairwise distances between training features
            # [N_train, N_train]
            d2 = torch.cdist(z, z).pow(2)

            # Prevent each sample from predicting itself
            d2.fill_diagonal_(float("inf"))

            weights = torch.softmax(
                -0.5 * d2 / 0.5**2,
                dim=1
            )

            # Flatten B0 maps if necessary
            b0_shape = self.training_data_b0.shape[1:]

            b0_flat = self.training_data_b0.reshape(
                self.training_data_b0.shape[0],
                -1
            )

            # KDE prediction
            predicted_b0 = weights @ b0_flat

            predicted_b0 = predicted_b0.reshape(
                self.training_data_b0.shape
            )

            loss = F.mse_loss(
                predicted_b0,
                self.training_data_b0
            )

            rmse = torch.sqrt(loss)

            loss.backward()
            self.optimizer.step()

            loss_history.append(loss.item())
            rmse_history.append(rmse.item())

            if epoch % 20 == 0:
                print(
                    f"{epoch:4d} | "
                    f"MSE = {loss.item():.6f} | "
                    f"RMSE = {rmse.item():.6f}"
                )

        # Plot AFTER training
        epochs = np.arange(len(loss_history))

        plt.figure()
        plt.plot(epochs, loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training MSE")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(epochs, rmse_history)
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Training RMSE")
        plt.grid()
        plt.show()

        return predicted_b0


    def test(self, KDE_object):

        self.model.eval()

        with torch.no_grad():

            # Training OCM -> learned reference features
            training_features = self.model(
                self.training_data_ocm
            )

            # Test OCM -> learned query features
            test_features = self.model(
                self.test_data_ocm
            )

            self.predicted_b0 = KDE_object.kde_predict(
                query_z=test_features,
                reference_z=training_features,
                reference_b0=self.training_data_b0,
                h=0.5
            )

            # MSE independently for each test sample
            dims = tuple(range(1, self.test_data_b0.ndim))

            test_mse = torch.mean(
                (self.predicted_b0 - self.test_data_b0) ** 2,
                dim=dims
            )

            test_rmse = torch.sqrt(test_mse)

            print("Mean test MSE:", test_mse.mean().item())
            print("Mean test RMSE:", test_rmse.mean().item())

            return test_rmse
    
    def save_b0_mat(self, filename):

        predicted_b0_np = (
            self.predicted_b0
            .detach()
            .cpu()
            .numpy()
        )

        scipy.io.savemat(
            filename,
            {"predicted_b0": predicted_b0_np}
        )

        print("Test B0:", self.test_data_b0.shape)
        print("Predicted B0:", self.predicted_b0.shape)


