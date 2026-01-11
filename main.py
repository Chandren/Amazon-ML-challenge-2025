"""
Run this file to execute the complete pipeline

Usage:
    python main.py                    # Run complete pipeline
    python main.py --skip-features    # Skip feature extraction (if already done)
    python main.py --skip-training    # Skip training (if models exist)
"""

# Disable TensorFlow to avoid tf-keras dependency issues
import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.feature_extraction import FeatureExtractor
from src.model_training import ModelTrainer
from src.prediction import Predictor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Amazon ML Challenge 2025 - Price Prediction'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature extraction (use existing features)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing models)'
    )
    parser.add_argument(
        '--only-test',
        action='store_true',
        help='Only process test data and generate predictions'
    )
    return parser.parse_args()

def main():
    """Main execution pipeline"""
    
    # ASCII art header
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            AMAZON ML CHALLENGE 2025                          ║
    ║            Smart Product Pricing                             ║
    ║                                                              ║
    ║            Multi-modal Ensemble Approach                     ║
    ║            (Text + Images → XGBoost + LightGBM)              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Parse arguments
    args = parse_args()
    
    # Initialize configuration
    print("\n🔧 Initializing configuration...")
    config = Config()
    
    # Create necessary directories
    config.create_directories()
    
    # Verify input files exist
    if not config.verify_input_files():
        print("\n Missing required input files. Please ensure train.csv and test.csv are in dataset/ folder.")
        return
    
    print("\n Setup complete! Starting pipeline...\n")
    
    try:
        # ==================== FEATURE EXTRACTION ====================
        if not args.skip_features and not args.only_test:
            print("\n" + "="*80)
            print("PHASE 1: FEATURE EXTRACTION")
            print("="*80)
            
            extractor = FeatureExtractor(config)
            
            # Process training data
            print("\n>>> Processing TRAINING data...")
            extractor.process_train_data()
            
            # Process test data
            print("\n>>> Processing TEST data...")
            extractor.process_test_data()
            
            print("\n Feature extraction complete!")
        else:
            print("\n⏭ Skipping feature extraction (using existing features)")
        
        # ==================== MODEL TRAINING ====================
        if not args.skip_training and not args.only_test:
            print("\n" + "="*80)
            print("PHASE 2: MODEL TRAINING")
            print("="*80)
            
            trainer = ModelTrainer(config)
            
            # Train with validation, then final training
            validation_smape = trainer.train_ensemble()
            
            print(f"\n Training complete! Validation SMAPE: {validation_smape:.2f}%")
        else:
            print("\n⏭ Skipping model training (using existing models)")
        
        # ==================== PREDICTION GENERATION ====================
        print("\n" + "="*80)
        print("PHASE 3: PREDICTION GENERATION")
        print("="*80)
        
        predictor = Predictor(config)
        submission = predictor.generate_submission()
        
        # ==================== FINAL SUMMARY ====================
        print("\n" + "="*80)
        print(" PIPELINE COMPLETE!")
        print("="*80)
        
        print("\n Output Files:")
        print(f"   test_out.csv - Predictions for submission")
        print(f"   models/ - Trained model files")
        print(f"   features/ - Extracted features")
        
        
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
