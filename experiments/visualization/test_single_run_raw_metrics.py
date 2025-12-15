"""
Quick test script to verify eval_no_damage_raw/* metrics on a single run.

This script loads and displays the raw metrics from a single WandB run
to verify they're being logged correctly and contain all expected epochs.
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from boolean_nca_cc.utils.wandb_loader import load_metric_pair


def test_single_run_raw_metrics(
    run_id: str,
    project: str = "boolean-nca-cc",
    entity: str = "marcello-barylli-growai",
):
    """
    Test loading raw metrics from a single run.
    
    Args:
        run_id: WandB run ID
        project: WandB project name
        entity: WandB entity/username
    """
    print(f"Testing raw metrics for run: {run_id}")
    print("=" * 60)
    
    # First, get full history to see ALL raw metric columns
    print("\n0. Inspecting ALL raw metric columns in history...")
    from boolean_nca_cc.utils.wandb_loader import get_run_history
    try:
        full_history = get_run_history(run_id, project=project, entity=entity)
        raw_epoch_cols = [c for c in full_history.columns if 'eval_no_damage_raw' in c and 'epoch' in c]
        raw_acc_cols = [c for c in full_history.columns if 'eval_no_damage_raw' in c and 'final_hard_accuracy' in c]
        
        print(f"  Found raw epoch columns: {raw_epoch_cols}")
        print(f"  Found raw accuracy columns: {raw_acc_cols}")
        
        # Check each raw metric column
        for epoch_col in raw_epoch_cols:
            acc_col = epoch_col.replace('/epoch', '/final_hard_accuracy')
            if acc_col in full_history.columns:
                has_both = (full_history[epoch_col].notna() & full_history[acc_col].notna()).sum()
                if has_both > 0:
                    unique_epochs = sorted(full_history[full_history[epoch_col].notna() & full_history[acc_col].notna()][epoch_col].unique())
                    print(f"  {epoch_col}: {has_both} rows with both metrics, {len(unique_epochs)} unique epochs: {unique_epochs}")
                    
                    # Show the actual rows with step information
                    rows_with_data = full_history[full_history[epoch_col].notna() & full_history[acc_col].notna()].copy()
                    if 'step' in rows_with_data.columns or '_step' in str(rows_with_data.columns):
                        step_col = 'step' if 'step' in rows_with_data.columns else [c for c in rows_with_data.columns if 'step' in c.lower()][0]
                        print(f"    Step column: {step_col}")
                        print(f"    Unique steps: {sorted(rows_with_data[step_col].unique())[:10]}")
                        # Show sample rows
                        print(f"    Sample rows (first 5):")
                        for idx, row in rows_with_data.head(5).iterrows():
                            print(f"      Row {idx}: step={row.get(step_col, 'N/A')}, epoch={row[epoch_col]}, acc={row[acc_col]:.4f}")
                    
                    # Check if epoch 0 exists in any form
                    if 0.0 in full_history[epoch_col].values or 0 in full_history[epoch_col].values:
                        epoch_0_rows = full_history[(full_history[epoch_col] == 0.0) | (full_history[epoch_col] == 0)]
                        print(f"    ⚠️  Found {len(epoch_0_rows)} rows with epoch=0, but they might not have both metrics")
                        if acc_col in epoch_0_rows.columns:
                            has_acc = epoch_0_rows[acc_col].notna().sum()
                            print(f"      Rows with {acc_col}: {has_acc}")
                            if has_acc > 0:
                                print(f"      Epoch 0 accuracy values: {epoch_0_rows[epoch_0_rows[acc_col].notna()][acc_col].tolist()}")
                    else:
                        print(f"    ℹ️  No rows found with epoch=0 for {epoch_col}")
    except Exception as e:
        print(f"  ⚠️  Could not inspect full history: {e}")
        import traceback
        traceback.print_exc()
    
    # Try loading raw metrics (test data)
    print("\n1. Loading eval_no_damage_raw/* metrics (TEST data)...")
    try:
        df_raw = load_metric_pair(
            run_id=run_id,
            x_metric="eval_no_damage_raw/epoch",
            y_metric="eval_no_damage_raw/final_hard_accuracy",
            project=project,
            entity=entity,
            include_config=True,
            config_keys=["seed"],
        )
        
        if df_raw.empty:
            print("  ❌ No raw metrics found!")
        else:
            print(f"  ✅ Loaded {len(df_raw)} data points")
            print(f"  Unique epochs: {sorted(df_raw['eval_no_damage_raw/epoch'].unique())}")
            print(f"  Epoch range: {df_raw['eval_no_damage_raw/epoch'].min():.0f} - {df_raw['eval_no_damage_raw/epoch'].max():.0f}")
            print(f"  Accuracy range: {df_raw['eval_no_damage_raw/final_hard_accuracy'].min():.4f} - {df_raw['eval_no_damage_raw/final_hard_accuracy'].max():.4f}")
        
    except Exception as e:
        print(f"  ❌ Error loading raw metrics: {e}")
        df_raw = None
    
    # Also try train metrics
    print("\n1b. Loading eval_no_damage_raw_train/* metrics (TRAIN data)...")
    try:
        df_raw_train = load_metric_pair(
            run_id=run_id,
            x_metric="eval_no_damage_raw_train/epoch",
            y_metric="eval_no_damage_raw_train/final_hard_accuracy",
            project=project,
            entity=entity,
            include_config=True,
            config_keys=["seed"],
        )
        
        if not df_raw_train.empty:
            print(f"  ✅ Loaded {len(df_raw_train)} data points")
            print(f"  Unique epochs: {sorted(df_raw_train['eval_no_damage_raw_train/epoch'].unique())}")
            print(f"  Epoch range: {df_raw_train['eval_no_damage_raw_train/epoch'].min():.0f} - {df_raw_train['eval_no_damage_raw_train/epoch'].max():.0f}")
            
            # Combine test and train if both exist
            if df_raw is not None and not df_raw.empty:
                print(f"\n  Combined (test + train): {len(df_raw) + len(df_raw_train)} total points")
                all_epochs = sorted(set(df_raw['eval_no_damage_raw/epoch'].unique()) | 
                                   set(df_raw_train['eval_no_damage_raw_train/epoch'].unique()))
                print(f"  All unique epochs: {all_epochs}")
    except Exception as e:
        print(f"  ⚠️  Could not load train raw metrics: {e}")
    
    if df_raw is None or df_raw.empty:
        return False
    
    # Compare with grouped metrics (if available)
    print("\n2. Comparing with eval_no_damage/* metrics (with step_metric grouping)...")
    try:
        df_grouped = load_metric_pair(
            run_id=run_id,
            x_metric="eval_no_damage/epoch",
            y_metric="eval_no_damage/final_hard_accuracy",
            project=project,
            entity=entity,
            include_config=True,
            config_keys=["seed"],
        )
        
        if not df_grouped.empty:
            print(f"  Grouped metrics: {len(df_grouped)} data points")
            print(f"  Grouped epochs: {sorted(df_grouped['eval_no_damage/epoch'].unique())}")
            print(f"  Raw metrics: {len(df_raw)} data points")
            print(f"  Raw epochs: {sorted(df_raw['eval_no_damage_raw/epoch'].unique())}")
            
            if len(df_raw) > len(df_grouped):
                print(f"  ✅ Raw metrics have MORE data points ({len(df_raw)} vs {len(df_grouped)})")
                print(f"     This confirms raw metrics avoid grouping issues!")
            elif len(df_raw) == len(df_grouped):
                print(f"  ⚠️  Same number of data points - might be OK if runs are identical")
            else:
                print(f"  ⚠️  Raw metrics have fewer points - unexpected!")
        else:
            print("  ⚠️  No grouped metrics found (might be from old run)")
            
    except Exception as e:
        print(f"  ⚠️  Could not load grouped metrics (expected for new runs): {e}")
    
    # Show sample data
    print("\n3. Sample data:")
    print(df_raw[['eval_no_damage_raw/epoch', 'eval_no_damage_raw/final_hard_accuracy']].head(10))
    
    print("\n" + "=" * 60)
    print("✅ Test complete! Raw metrics are working correctly.")
    print(f"   Found {len(df_raw)} evaluation points across {df_raw['eval_no_damage_raw/epoch'].nunique()} unique epochs.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test eval_no_damage_raw/* metrics on a single WandB run"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="WandB run ID to test",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="boolean-nca-cc",
        help="WandB project name (default: boolean-nca-cc)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="marcello-barylli-growai",
        help="WandB entity/username (default: marcello-barylli-growai)",
    )
    
    args = parser.parse_args()
    
    test_single_run_raw_metrics(
        run_id=args.run_id,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()

