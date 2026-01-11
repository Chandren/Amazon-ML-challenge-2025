#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.feature_extraction import FeatureExtractor

def main():
    
    # Initialize configuration
    print("\n Initializing configuration...")
    config = Config()
    
    # Create necessary directories
    config.create_directories()
    
    # Initialize feature extractor
    extractor = FeatureExtractor(config)
    
    try:
        # Step 1: Complete image download for training data
        print("\n" + "="*80)
        print("STEP 1: COMPLETING IMAGE DOWNLOAD (TRAIN)")
        print("="*80)
        
        train_csv = config.TRAIN_CSV
        extractor.download_images_for_dataset(train_csv, config.TRAIN_IMAGES_DIR)
        
        # Step 2: Extract ResNet-50 features
        print("\n" + "="*80)
        print("STEP 2: EXTRACTING ResNet-50 FEATURES (TRAIN)")
        print("="*80)
        
        extractor.extract_image_features_resnet(
            train_csv,
            config.TRAIN_IMAGES_DIR,
            config.TRAIN_IMAGE_FEATURES,
            batch_size=32,  # Process 32 images at once
            checkpoint_every=5000  # Save checkpoint every 5000 images
        )
        
        # Step 3: Extract image statistics
        print("\n" + "="*80)
        print("STEP 3: EXTRACTING IMAGE STATISTICS (TRAIN)")
        print("="*80)
        
        train_stats_path = config.FEATURES_DIR / 'train_image_stats.npy'
        extractor.extract_image_statistics(
            train_csv,
            config.TRAIN_IMAGES_DIR,
            train_stats_path,
            checkpoint_every=10000
        )
        
        # Step 4: Process test data
        print("\n" + "="*80)
        print("STEP 4: PROCESSING TEST DATA")
        print("="*80)
        
        test_csv = config.TEST_CSV
        
        # Download test images
        print("\n>>> Downloading test images...")
        extractor.download_images_for_dataset(test_csv, config.TEST_IMAGES_DIR)
        
        # Extract ResNet features for test
        print("\n>>> Extracting ResNet-50 features (test)...")
        extractor.extract_image_features_resnet(
            test_csv,
            config.TEST_IMAGES_DIR,
            config.TEST_IMAGE_FEATURES,
            batch_size=32,
            checkpoint_every=5000
        )
        
        # Extract image statistics for test
        print("\n>>> Extracting image statistics (test)...")
        test_stats_path = config.FEATURES_DIR / 'test_image_stats.npy'
        extractor.extract_image_statistics(
            test_csv,
            config.TEST_IMAGES_DIR,
            test_stats_path,
            checkpoint_every=10000
        )
        
        print("\n" + "="*80)
        print(" IMAGE PROCESSING COMPLETE!")
        print("="*80)
        
        print("\n What's been completed:")
        print("    Training images downloaded")
        print("    Training ResNet-50 features extracted")
        print("    Training image statistics extracted")
        print("    Test images downloaded")
        print("    Test ResNet-50 features extracted")
        print("    Test image statistics extracted")
        
        print("\n Next steps:")
        print("   1. Run model training:")
        print("      ./venv/bin/python -c \"from src.model_training import ModelTrainer; from src.config import Config; trainer = ModelTrainer(Config()); trainer.train_ensemble()\"")
        print("\n   2. Or run the complete pipeline:")
        print("      ./run.sh --skip-features")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n Process interrupted by user")
        print("   Don't worry! Progress has been saved in checkpoints.")
        print("   Run this script again to resume from where you left off.")
        return 130
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n   Progress has been saved in checkpoints.")
        print("   Run this script again to resume.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
