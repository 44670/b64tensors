#!/usr/bin/env python3
"""
Basic usage example for b64tensors package.

This script demonstrates how to encode and decode PyTorch tensors
using the b64tensors library.
"""

import torch
import b64tensors


def main():
    print("=== b64tensors Basic Usage Example ===\n")
    
    # Example 1: Basic tensor encoding/decoding
    print("1. Basic tensor encoding/decoding:")
    tensor = torch.randn(5, 10, dtype=torch.float32)
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Original tensor dtype: {tensor.dtype}")
    
    # Encode the tensor
    encoded = b64tensors.encode(tensor)
    print(f"Encoded string length: {len(encoded)} characters")
    print(f"Encoded format: {encoded[:50]}...")  # Show first 50 characters
    
    # Decode the tensor
    decoded = b64tensors.decode(encoded)
    print(f"Decoded tensor shape: {decoded.shape}")
    print(f"Decoded tensor dtype: {decoded.dtype}")
    print(f"Tensors are equal: {torch.equal(tensor, decoded)}")
    print()
    
    # Example 2: Working with different data types
    print("2. Different data types:")
    test_tensors = [
        ("float32", torch.randn(3, 3, dtype=torch.float32)),
        ("float16", torch.randn(3, 3, dtype=torch.float16)),
        ("int32", torch.randint(0, 100, (3, 3), dtype=torch.int32)),
        ("bool", torch.randint(0, 2, (3, 3), dtype=torch.bool)),
    ]
    
    for name, test_tensor in test_tensors:
        encoded = b64tensors.encode(test_tensor)
        decoded = b64tensors.decode(encoded)
        print(f"{name}: {torch.equal(test_tensor, decoded)} (shape: {test_tensor.shape})")
    print()
    
    # Example 3: Dictionary encoding/decoding
    print("3. Dictionary encoding/decoding:")
    tensor_dict = {
        "weights": torch.randn(10, 20, dtype=torch.float32),
        "biases": torch.randn(20, dtype=torch.float32),
        "labels": torch.randint(0, 10, (100,), dtype=torch.int32)
    }
    
    print(f"Original dict keys: {list(tensor_dict.keys())}")
    for key, tensor in tensor_dict.items():
        print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # Encode the dictionary
    encoded_dict = b64tensors.encode_dict(tensor_dict)
    print(f"Encoded dict keys: {list(encoded_dict.keys())}")
    
    # Decode the dictionary
    decoded_dict = b64tensors.decode_dict(encoded_dict)
    print(f"Decoded dict keys: {list(decoded_dict.keys())}")
    
    # Verify all tensors are equal
    all_equal = all(torch.equal(tensor_dict[key], decoded_dict[key]) 
                   for key in tensor_dict.keys())
    print(f"All tensors equal after round-trip: {all_equal}")
    print()
    
    # Example 4: Handling edge cases
    print("4. Edge cases:")
    
    # Scalar tensor
    scalar = torch.tensor(42.0, dtype=torch.float32)
    scalar_encoded = b64tensors.encode(scalar)
    scalar_decoded = b64tensors.decode(scalar_encoded)
    print(f"Scalar tensor (shape {scalar.shape}): {torch.equal(scalar, scalar_decoded)}")
    
    # Empty tensor
    empty = torch.empty(0, dtype=torch.float32)
    empty_encoded = b64tensors.encode(empty)
    empty_decoded = b64tensors.decode(empty_encoded)
    print(f"Empty tensor (shape {empty.shape}): {torch.equal(empty, empty_decoded)}")
    
    # Large tensor
    large = torch.randn(1000, 1000, dtype=torch.float32)
    large_encoded = b64tensors.encode(large)
    large_decoded = b64tensors.decode(large_encoded)
    print(f"Large tensor (shape {large.shape}): {torch.equal(large, large_decoded)}")
    print(f"Large tensor encoded size: {len(large_encoded)} characters")
    print()
    
    # Example 5: Error handling
    print("5. Error handling:")
    
    try:
        # Invalid format
        b64tensors.decode("invalid_format")
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        # Unsupported dtype
        complex_tensor = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
        b64tensors.encode(complex_tensor)
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main() 