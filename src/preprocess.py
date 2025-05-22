import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_data(file_path):
    """Load RadioML dataset."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def augment_iq_data(X_data, theta_rad):
    """
    Augment I/Q data by rotating the I and Q channels.
    Args:
        X_data: Input data of shape (num_samples, 2, sequence_length)
        theta_rad: Rotation angle in radians
    Returns:
        Augmented data of the same shape as X_data
    """
    I_original = X_data[:, 0, :]
    Q_original = X_data[:, 1, :]
    
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    I_augmented = I_original * cos_theta - Q_original * sin_theta
    Q_augmented = I_original * sin_theta + Q_original * cos_theta
    
    X_augmented = np.stack((I_augmented, Q_augmented), axis=1)
    return X_augmented


def prepare_data(dataset, test_size=0.2, validation_split=0.1, snrs_filter=None, 
                 augment_data=False):
    """
    Prepare data for training and testing.
    
    Args:
        dataset: The loaded RadioML dataset
        test_size: Proportion of data to use for testing
        validation_split: Proportion of training data to use for validation
        snrs_filter: List of SNR values to include (None=all)
        augment_data: Boolean flag to enable/disable data augmentation on training set
        
    Returns:
        X_train, X_val, X_test: Training, validation and test data
        y_train, y_val, y_test: Training, validation and test labels
        classes: List of modulation types
    """
    # Get the list of modulation types
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    
    # Filter by SNR if specified
    if snrs_filter is None:
        snrs_to_process = sorted(list(set([k[1] for k in dataset.keys()])))
    else:
        snrs_to_process = snrs_filter
    
    # Create a mapping from modulation type to index
    mod_to_index = {mod: i for i, mod in enumerate(mods)}
    
    # Lists to hold the samples and labels
    X_list = []
    y_list = []
    
    # Collect all samples
    for mod in mods:
        for snr_val in snrs_to_process:
            key = (mod, snr_val)
            if key in dataset:
                X_list.append(dataset[key])
                y_list.append(np.ones(len(dataset[key])) * mod_to_index[mod])
    
    # Convert lists to numpy arrays
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list).astype(int)
    
    # Split data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=42, stratify=y_all)
    
    # Further split training data into training and validation sets
    # Adjust validation_split calculation if test_size is 0
    if 1 - test_size == 0: # Avoid division by zero if test_size is 1.0
        val_size_adjusted = 0
    else:
        val_size_adjusted = validation_split / (1 - test_size)

    if val_size_adjusted >= 1.0: # Ensure val_size_adjusted is less than 1
        val_size_adjusted = 0.5 # Or some other sensible default if validation_split is too high for the remaining data
        print(f"Warning: validation_split too high for remaining data after test split. Adjusted val_size to {val_size_adjusted}")

    if val_size_adjusted > 0 and X_train_val.shape[0] > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val)
    else: # If val_size_adjusted is 0 or X_train_val is empty
        X_train, y_train = X_train_val, y_train_val
        # Initialize X_val, y_val as empty arrays with correct number of dimensions if X_train is not empty
        if X_train.ndim == 3: # For X data (num_samples, 2, sequence_length)
             X_val = np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]) if X_train.size > 0 else np.array([]).reshape(0,2,0) # handle case where X_train could be (0,2,128)
        elif X_train.ndim == 2: # For X data (num_samples, features)
             X_val = np.array([]).reshape(0, X_train.shape[1]) if X_train.size > 0 else np.array([]).reshape(0,0)
        else: # Fallback for 1D or other
             X_val = np.array([])
        y_val = np.array([])


    # Data Augmentation for training set
    if augment_data and X_train.shape[0] > 0:
        print(f"Starting data augmentation: 11 rotations, each by 30 degree increments.")
        X_original_for_aug = X_train.copy()
        y_original_for_aug = y_train.copy()
        
        augmented_X_accumulated = []
        augmented_y_accumulated = []
        
        for i in range(11): # 0 to 10 for 11 augmentations
            current_angle_deg = (i + 1) * 30.0
            print(f"Augmenting training data: rotation {i+1}/11, angle: {current_angle_deg} degrees.")
            theta_rad = np.deg2rad(current_angle_deg)
            X_augmented_single = augment_iq_data(X_original_for_aug, theta_rad)
            
            augmented_X_accumulated.append(X_augmented_single)
            augmented_y_accumulated.append(y_original_for_aug) # Append original labels for this augmented set
            
        if augmented_X_accumulated:
            X_train = np.concatenate([X_train] + augmented_X_accumulated, axis=0)
            y_train = np.concatenate([y_train] + augmented_y_accumulated, axis=0)
        
        print(f"Size of training set before augmentation: {X_original_for_aug.shape[0]}")
        print(f"Number of augmentations performed: 11")
        print(f"Size of training set after augmentation: {X_train.shape[0]}")

    # Convert labels to one-hot encoding
    num_classes = len(mods)
    if y_train.size > 0: # Check if y_train is not empty before one-hot encoding
        y_train = to_categorical(y_train, num_classes)
    else: # Handle empty y_train
        y_train = np.array([]).reshape(0, num_classes)

    if y_val.size > 0: # Check if y_val is not empty
        y_val = to_categorical(y_val, num_classes)
    else: # Handle empty y_val
        y_val = np.array([]).reshape(0, num_classes)

    if y_test.size > 0: # Check if y_test is not empty
        y_test = to_categorical(y_test, num_classes)
    else: # Handle empty y_test
        y_test = np.array([]).reshape(0, num_classes)

    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, mods


def prepare_data_by_snr(dataset, test_size=0.2, validation_split=0.1, specific_snrs=None,
                        augment_data=False):
    """
    Organize data for training and testing, keeping samples separated by SNR.
    Useful for evaluating performance across different SNRs.
    Can also augment training data.
    
    Args:
        dataset: The loaded RadioML dataset
        test_size: Proportion of data to use for testing
        validation_split: Proportion of training data to use for validation
        specific_snrs: List of SNR values to include (None=all)
        augment_data: Boolean flag to enable/disable data augmentation on training set

    Returns:
        X_train, X_val, X_test: Training, validation and test data
        y_train, y_val, y_test: Training, validation and test labels
        snr_train, snr_val, snr_test: SNR values for each sample
        classes: List of modulation types
    """
    # Get the list of modulation types and SNRs
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    if specific_snrs is None:
        snrs_list = sorted(list(set([k[1] for k in dataset.keys()])))
    else:
        snrs_list = specific_snrs
        
    # Create a mapping from modulation type to index
    mod_to_index = {mod: i for i, mod in enumerate(mods)}
    
    # Lists to hold the samples, labels, and SNR values
    X_all_list = []
    y_all_list = []
    snr_values_all_list = []
    
    # Collect all samples
    for mod in mods:
        for snr_val in snrs_list:
            key = (mod, snr_val)
            if key in dataset:
                X_all_list.append(dataset[key])
                y_all_list.append(np.ones(len(dataset[key])) * mod_to_index[mod])
                snr_values_all_list.append(np.ones(len(dataset[key])) * snr_val)
    
    # Convert lists to numpy arrays
    X_all = np.vstack(X_all_list)
    y_all = np.hstack(y_all_list).astype(int)
    snr_values_all = np.hstack(snr_values_all_list)
    
    # Split data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test, snr_train_val, snr_test = train_test_split(
        X_all, y_all, snr_values_all, test_size=test_size, random_state=42, stratify=y_all
    )
    
    # Further split training data into training and validation sets
    if 1 - test_size == 0: 
        val_size_adjusted = 0
    else:
        val_size_adjusted = validation_split / (1 - test_size)
    
    if val_size_adjusted >= 1.0: 
        val_size_adjusted = 0.5 
        print(f"Warning: validation_split too high for remaining data after test split. Adjusted val_size to {val_size_adjusted}")

    if val_size_adjusted > 0 and X_train_val.shape[0] > 0 :
        X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
            X_train_val, y_train_val, snr_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val
        )
    else: 
        X_train, y_train, snr_train = X_train_val, y_train_val, snr_train_val
        if X_train.ndim == 3:
             X_val = np.array([]).reshape(0, X_train.shape[1], X_train.shape[2]) if X_train.size > 0 else np.array([]).reshape(0,2,0)
        elif X_train.ndim == 2:
             X_val = np.array([]).reshape(0, X_train.shape[1]) if X_train.size > 0 else np.array([]).reshape(0,0)
        else: 
             X_val = np.array([])
        y_val = np.array([]) 
        snr_val = np.array([])


    # Data Augmentation for training set
    if augment_data and X_train.shape[0] > 0:
        print(f"Starting data augmentation for SNR-specific data: 11 rotations, each by 30 degree increments.")
        X_original_for_aug = X_train.copy()
        y_original_for_aug = y_train.copy()
        snr_original_for_aug = snr_train.copy()

        augmented_X_accumulated = []
        augmented_y_accumulated = []
        augmented_snr_accumulated = []

        angle = 90
        num = 360 // angle - 1
        for i in range(num): # 0 to 6 for 7 augmentations
            current_angle_deg = (i + 1) * angle
            print(f"Augmenting training data (SNR-specific): rotation {i+1}/{num}, angle: {current_angle_deg} degrees.")
            theta_rad = np.deg2rad(current_angle_deg)
            X_augmented_single = augment_iq_data(X_original_for_aug, theta_rad)
            
            augmented_X_accumulated.append(X_augmented_single)
            augmented_y_accumulated.append(y_original_for_aug)
            augmented_snr_accumulated.append(snr_original_for_aug)
            
        if augmented_X_accumulated:
            X_train = np.concatenate([X_train] + augmented_X_accumulated, axis=0)
            y_train = np.concatenate([y_train] + augmented_y_accumulated, axis=0)
            snr_train = np.concatenate([snr_train] + augmented_snr_accumulated, axis=0)

        print(f"Size of training set before augmentation (SNR-specific): {X_original_for_aug.shape[0]}")
        print(f"Number of augmentations performed: 7")
        print(f"Size of training set after augmentation (SNR-specific): {X_train.shape[0]}")

    # Convert labels to one-hot encoding
    num_classes = len(mods)
    if y_train.size > 0:
        y_train = to_categorical(y_train, num_classes)
    else:
        y_train = np.array([]).reshape(0, num_classes)

    if y_val.size > 0:
        y_val = to_categorical(y_val, num_classes)
    else: 
        y_val = np.array([]).reshape(0, num_classes)

    if y_test.size > 0:
        y_test = to_categorical(y_test, num_classes)
    else:
        y_test = np.array([]).reshape(0, num_classes)

    print(f"Training set: {X_train.shape}, {y_train.shape}, SNR array: {snr_train.shape if snr_train.size > 0 else 'empty'}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}, SNR array: {snr_val.shape if snr_val.size > 0 else 'empty'}")
    print(f"Test set: {X_test.shape}, {y_test.shape}, SNR array: {snr_test.shape if snr_test.size > 0 else 'empty'}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods


if __name__ == "__main__":
    # This can be used for testing the preprocessing functions
    file_path = "../RML2016.10a_dict.pkl" 
    
    try:
        dataset = load_data(file_path)
        print("Dataset loaded successfully for __main__ test.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path} when running preprocess.py directly.")
        print("Please ensure the path is correct relative to the script's execution directory.")
        dataset = None

    if dataset:
        # Basic preprocessing
        print("\nTesting prepare_data:")
        X_train, X_val, X_test, y_train, y_val, y_test, mods = prepare_data(dataset, augment_data=False)
        print("\nTesting prepare_data with augmentation:")
        X_train_aug, _, _, y_train_aug, _, _, _ = prepare_data(dataset, augment_data=True)
        
        # SNR-aware preprocessing
        print("\nTesting prepare_data_by_snr:")
        X_train_snr, X_val_snr, X_test_snr, y_train_snr, y_val_snr, y_test_snr, snr_train, snr_val, snr_test, mods_snr = prepare_data_by_snr(dataset, augment_data=False)
        print("\nTesting prepare_data_by_snr with augmentation:")
        X_train_snr_aug, _, _, y_train_snr_aug, _, _, snr_train_aug, _, _, _ = prepare_data_by_snr(dataset, augment_data=True)