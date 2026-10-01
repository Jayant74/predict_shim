from utils.dataloader import dataloader


if __name__ ==  "__main__":

    # Initialize loader and find matching OCM/B0 scans
    data = dataloader()

    print("Matched datasets:")
    print(data.io_match)

    # Generate OCM windows and corresponding B0 datasets
    data.generate_datasets()

    # Example:
    # scan 1 = training
    # scan 3 = testing
    train_idx = 1
    test_idx = 3
    print(f"Training Scan {train_idx + 1} and Test Scan {test_idx + 1}")

    data.training_data(train_idx)
    data.test_data(test_idx)
    data.flatten_data()

    print(data.training_data_ocm.shape)
    print(data.training_data_b0.shape)
    print(data.Ut.shape)
    print(data.UT.shape)

