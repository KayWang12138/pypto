"""
Bundled helper for golden vs PyPTO output comparison.
Import from kernel validation runners: see skills/pypto-op-validate/SKILL.md.
"""
import torch


def detailed_tensor_compare(tensor1, tensor2, tensor_name, rtol=1e-3, atol=1e-3, verbose=True, max_outliers_display=20):
    """
    Detailed tensor comparison, analyzing the proportion of elements that are out of tolerance, and displaying specific information about those that exceed the tolerance.

    Args:
    tensor1: The first tensor.
    tensor2: The second tensor.
    rtol: Relative tolerance.
    atol: Absolute tolerance.
    verbose: Whether to print detailed information.
    max_outliers_display: Maximum number of out-of-tolerance elements to display.

    Returns:
    dict: A dictionary containing the comparison results.
    """
    # Ensure tensors are comparable
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # Calculate the difference
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)

    # Tolerance Check
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # Statistics
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # Difference Statistics
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()

        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        sorted_indices = torch.argsort(outlier_diffs, descending=True)
        sorted_outlier_indices = tuple(ind[sorted_indices] for ind in outlier_indices)
        sorted_outlier_values1 = outlier_values1[sorted_indices]
        sorted_outlier_values2 = outlier_values2[sorted_indices]
        sorted_outlier_diffs = outlier_diffs[sorted_indices]
        sorted_outlier_relative_diffs = outlier_relative_diffs[sorted_indices]

    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        sorted_outlier_indices = None
        sorted_outlier_values1 = None
        sorted_outlier_values2 = None
        sorted_outlier_diffs = None
        sorted_outlier_relative_diffs = None

    result = {
        'total_elements': total_elements,
        'out_of_tolerance_count': out_of_tolerance_count,
        'out_of_tolerance_ratio': out_of_tolerance_ratio,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_out_of_tolerance_diff': max_out_diff,
        'mean_out_of_tolerance_diff': mean_out_diff,
        'all_close': out_of_tolerance_count == 0,
        'tolerance_mask': tolerance_mask,
        'diff_tensor': diff,
        'outlier_indices': sorted_outlier_indices,
        'outlier_values1': sorted_outlier_values1,
        'outlier_values2': sorted_outlier_values2,
        'outlier_diffs': sorted_outlier_diffs,
        'outlier_relative_diffs': sorted_outlier_relative_diffs
    }

    if verbose:
        print("\n" + "="*60)
        print("📊 Tensor Detailed Comparison Report")
        print(f"name: {tensor_name}")
        print("="*60)
        print(f"Total number of elements: {total_elements:,}")
        print(f"Number of elements exceeding tolerance: {out_of_tolerance_count:,}")
        print(f"Out of Tolerance Ratio: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio*100:.4f}%)")
        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Average difference: {mean_diff:.6f}")
        print(f"Difference Standard Deviation: {std_diff:.6f}")
        print(f"Tolerance Settings: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            print(f"Maximum deviation exceeding tolerance: {max_out_diff:.6f}")
            print(f"Average deviation exceeding tolerance: {mean_out_diff:.6f}")

            print(f"\n🔍 Details of elements exceeding tolerance limits (Before Displaying{min(max_outliers_display, out_of_tolerance_count)}):")
            print("-" * 80)
            print(f"{'Index':<20} {'Tensor1 value':<15} {'Tensor2 value':<15} {'Absolute difference':<12} {'Relative difference':<12}")
            print("-" * 80)

            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(tuple(sorted_outlier_indices[j][i].item() for j in range(len(sorted_outlier_indices))))
                print(f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} {sorted_outlier_values2[i].item():<15.6f} "
                      f"{sorted_outlier_diffs[i].item():<12.6f} {sorted_outlier_relative_diffs[i].item():<12.6f}")

            if out_of_tolerance_count > max_outliers_display:
                print(f"... And also {out_of_tolerance_count - max_outliers_display} An element exceeding the tolerance is not displayed.")

        print(f"\n✅ Tensor Matching: {result['all_close']}")
        print("="*60)

    return result
