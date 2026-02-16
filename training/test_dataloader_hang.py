"""Test to identify where the hang occurs."""
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from dataset.violation_dataset import load_violation_dataset

def test_single_batch():
    """Test loading a single batch."""
    logger.info("Loading dataset...")
    full_dataset = load_violation_dataset(
        data_root=Path('/home/satriabw/Track2Data'),
        split='train',
        context_frames=16,
        num_frames=32,
        img_size=(224, 224)
    )
    
    # Get 8 samples from video_001
    video_001_indices = [i for i in range(len(full_dataset)) 
                        if full_dataset.labels[i].video_id == 'video_001']
    indices = video_001_indices[:8]
    dataset = Subset(full_dataset, indices)
    
    logger.info(f"Testing with {len(dataset)} samples")
    
    # Test 1: Direct access
    logger.info("\n=== Test 1: Direct Dataset Access ===")
    try:
        sample = dataset[0]
        logger.info(f"✓ Sample 0 loaded: frames {sample['frames'].shape}, traj {sample['trajectory'].shape}")
    except Exception as e:
        logger.error(f"✗ Failed to load sample 0: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: DataLoader with num_workers=0
    logger.info("\n=== Test 2: DataLoader num_workers=0 ===")
    try:
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        logger.info("DataLoader created, fetching first batch...")
        
        for i, batch in enumerate(loader):
            logger.info(f"✓ Batch {i}: frames {batch['frames'].shape}, labels {batch['label']}")
            if i >= 1:  # Test 2 batches
                break
        logger.info("✓ DataLoader test passed")
    except Exception as e:
        logger.error(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Second iteration (this is where hangs occur)
    logger.info("\n=== Test 3: Second Iteration (Critical Test) ===")
    try:
        logger.info("Starting second iteration...")
        for i, batch in enumerate(loader):
            logger.info(f"✓ Iteration 2, Batch {i}: frames {batch['frames'].shape}")
            if i >= 1:
                break
        logger.info("✓ Second iteration passed!")
    except Exception as e:
        logger.error(f"✗ Second iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("\n=== All Tests Passed! ===")
    return True

if __name__ == '__main__':
    import time
    start = time.time()
    success = test_single_batch()
    elapsed = time.time() - start
    
    if success:
        logger.info(f"\n✓ SUCCESS - Completed in {elapsed:.1f}s")
    else:
        logger.error(f"\n✗ FAILED after {elapsed:.1f}s")
        sys.exit(1)
