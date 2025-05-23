import unittest
import numpy as np
import sys
import os
import tempfile
from tensorflow.keras.models import Sequential, Model # Added Model
from tensorflow.keras.layers import Input, Conv1D, Dense # Added Dense
from tensorflow.keras.optimizers import SGD # Added SGD

# Add src directory to Python path to allow direct import of modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import apply_gp_regression, apply_wavelet_denoising, apply_ddae_denoising
# RBF, Matern, RationalQuadratic are used internally by apply_gp_regression,
# so direct import for testing them separately is not strictly needed here
# unless we want to test kernel objects themselves.
# Attempting to import build_dae_model and compile_dae_model for the dummy model
try:
    from src.model.autoencoder_model import build_dae_model, compile_dae_model
    dae_builder_available = True
except ImportError:
    dae_builder_available = False
    print("Warning: src.model.autoencoder_model components not found. DDAE functional test will use a simpler dummy model.")


class TestDenoisingMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests in the class."""
        # Create a dummy DAE model once for all DDAE tests
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.dummy_model_path = os.path.join(cls.temp_dir.name, "dummy_dae_model.keras")
        
        if dae_builder_available:
            print("Using build_dae_model for dummy DAE model in tests.")
            # Use the actual DAE builder if available for more realistic structure
            dummy_model = build_dae_model(input_shape=(128, 2))
            compile_dae_model(dummy_model, learning_rate=0.01) # Compile with a basic optimizer
        else:
            print("Using simple Sequential model for dummy DAE model in tests.")
            # Fallback to a very simple model if the main DAE builder is not found
            dummy_model = Sequential([
                Input(shape=(128, 2)),
                Conv1D(filters=4, kernel_size=3, activation='relu', padding='same'), # Minimal processing
                Conv1D(filters=2, kernel_size=3, activation='linear', padding='same') # Output 2 channels
            ])
            dummy_model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')

        # "Train" for one epoch on random data
        # For DDAE, input and output are typically the same for training a simple autoencoder
        # or input is noisy and output is clean for a denoising autoencoder.
        # For this dummy test, we just need it to be able to predict.
        dummy_data_x = np.random.rand(10, 128, 2)
        dummy_data_y = np.random.rand(10, 128, 2) # Or use dummy_data_x if it's a simple AE
        dummy_model.fit(dummy_data_x, dummy_data_y, epochs=1, verbose=0)
        dummy_model.save(cls.dummy_model_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in the class."""
        cls.temp_dir.cleanup()

    def setUp(self):
        """Set up common test data."""
        np.random.seed(42) # for reproducibility
        self.sample_signal_length = 128
        self.sample_signal = np.random.rand(self.sample_signal_length) + 1j * np.random.rand(self.sample_signal_length)
        self.noise_std = 0.1

    def test_gp_regression_rbf(self):
        """Test GPR with RBF kernel."""
        denoised_signal = apply_gp_regression(
            self.sample_signal,
            self.noise_std,
            kernel_name='rbf',
            length_scale=1.0
        )
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))

    def test_gp_regression_matern(self):
        """Test GPR with Matern kernel."""
        denoised_signal = apply_gp_regression(
            self.sample_signal,
            self.noise_std,
            kernel_name='matern',
            length_scale=1.0,
            matern_nu=1.5
        )
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))

    def test_gp_regression_rational_quadratic(self):
        """Test GPR with RationalQuadratic kernel."""
        denoised_signal = apply_gp_regression(
            self.sample_signal,
            self.noise_std,
            kernel_name='rational_quadratic',
            length_scale=1.0,
            rational_quadratic_alpha=1.0
        )
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))

    def test_gp_regression_unknown_kernel(self):
        """Test GPR with an unknown kernel, expecting fallback to RBF."""
        # This test assumes that a warning is printed and RBF is used.
        # We can't directly test the print output without more complex mocking,
        # but we can ensure it still runs and produces valid output.
        denoised_signal = apply_gp_regression(
            self.sample_signal,
            self.noise_std,
            kernel_name='unknown_kernel_test',
            length_scale=1.0
        )
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))


    def test_wavelet_denoising(self):
        """Test Wavelet Denoising with default parameters."""
        denoised_signal = apply_wavelet_denoising(self.sample_signal)
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))
        # Check that the output is not the same as input (i.e., some denoising occurred)
        # This might not always be true if the signal is pure noise and thresholding removes everything
        # or if the signal is perfectly clean. Given random input, it should differ.
        self.assertFalse(np.array_equal(denoised_signal, self.sample_signal), "Wavelet denoising did not alter the signal.")


    def test_wavelet_denoising_params(self):
        """Test Wavelet Denoising with specific parameters."""
        denoised_signal = apply_wavelet_denoising(
            self.sample_signal,
            wavelet='db1', # Daubechies 1 (Haar)
            level=2,
            mode='hard'
        )
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))
        self.assertFalse(np.array_equal(denoised_signal, self.sample_signal), "Wavelet denoising (db1, level 2, hard) did not alter the signal.")


    def test_ddae_denoising_placeholder(self):
        """Test the DDAE Denoising placeholder function."""
        # Since it's a placeholder, it should return the original signal.
        # We can also capture stdout to check the message, but that's optional.
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            denoised_signal = apply_ddae_denoising(self.sample_signal)
        
        # Check console output
        self.assertIn("DDAE denoising is not yet implemented.", f.getvalue())

        # Check that the output is identical to the input
        self.assertIsInstance(denoised_signal, np.ndarray)
        self.assertEqual(denoised_signal.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal))
        self.assertTrue(np.array_equal(denoised_signal, self.sample_signal), "DDAE placeholder did not return the original signal.")

    def test_ddae_denoising_functional(self):
        """Test the DDAE Denoising function with a dummy model."""
        # Test Case 1: Model Not Found
        invalid_path = "non_existent_dummy_model.keras"
        # Ensure the invalid path truly doesn't exist for a clean test
        if os.path.exists(invalid_path):
            os.remove(invalid_path) 

        denoised_signal_no_model = apply_ddae_denoising(self.sample_signal, model_path=invalid_path)
        self.assertTrue(
            np.array_equal(denoised_signal_no_model, self.sample_signal),
            "DDAE should return original signal if model is not found."
        )

        # Test Case 2: Functional Denoising with Dummy Model
        # The dummy_model_path is created in setUpClass
        denoised_signal_functional = apply_ddae_denoising(self.sample_signal, model_path=self.dummy_model_path)
        
        self.assertIsInstance(denoised_signal_functional, np.ndarray)
        self.assertEqual(denoised_signal_functional.shape, self.sample_signal.shape)
        self.assertTrue(np.iscomplexobj(denoised_signal_functional))
        
        # Check if the signal was altered by the dummy model's processing
        # This assumes the dummy model (even if trivial) and norm/denorm will change the values.
        self.assertFalse(
            np.array_equal(denoised_signal_functional, self.sample_signal),
            "DDAE functional test with dummy model did not alter the signal as expected."
        )

    def test_ddae_denoising_zero_signal(self):
        """Test DDAE with a zero signal to check instance_max_val handling."""
        zero_signal = np.zeros_like(self.sample_signal)
        denoised_zero_signal = apply_ddae_denoising(zero_signal, model_path=self.dummy_model_path)
        self.assertTrue(
            np.array_equal(denoised_zero_signal, zero_signal),
            "DDAE should return the original zero signal if instance_max_val is 0."
        )


if __name__ == '__main__':
    unittest.main()
