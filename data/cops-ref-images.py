import json
import os
import shutil
from pathlib import Path
import subprocess
import math

def extract_cops_ref_image_ids(cops_ref_json_path):
    """Extract all unique image IDs from Cops-Ref dataset"""
    print("Extracting image IDs from Cops-Ref dataset...")
    
    with open(cops_ref_json_path, 'r') as f:
        data = json.load(f)
    
    image_ids = set()
    
    if isinstance(data, dict) and 'refs' in data:
        refs = data['refs']
    elif isinstance(data, list):
        refs = data
    else:
        raise ValueError("Unexpected JSON structure")
    
    for ref in refs:
        if ref is not None and 'imageId' in ref:
            image_ids.add(ref['imageId'])
    
    return sorted(list(image_ids))

def copy_cops_ref_images_in_batches(image_ids, local_gqa_dir, output_dir="./cops_ref_images", batch_size=100):
    """Copy Cops-Ref images organized in batches"""
    print(f"Copying {len(image_ids)} Cops-Ref images in batches of {batch_size}...")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of batches
    num_batches = math.ceil(len(image_ids) / batch_size)
    print(f"Creating {num_batches} batches...")
    
    copied_count = 0
    missing_count = 0
    batch_info = []
    
    for batch_num in range(num_batches):
        # Create batch directory
        batch_dir = os.path.join(output_dir, f"batch_{batch_num + 1}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Get image IDs for this batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(image_ids))
        batch_image_ids = image_ids[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_num + 1}/{num_batches} ({len(batch_image_ids)} images)...")
        
        batch_copied = 0
        batch_missing = 0
        
        for image_id in batch_image_ids:
            # GQA images are typically named {image_id}.jpg
            src_path = os.path.join(local_gqa_dir, f"{image_id}.jpg")
            dst_path = os.path.join(batch_dir, f"{image_id}.jpg")
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                batch_copied += 1
            else:
                print(f"Warning: Image {image_id}.jpg not found")
                missing_count += 1
                batch_missing += 1
        
        batch_info.append({
            'batch_num': batch_num + 1,
            'batch_dir': batch_dir,
            'copied': batch_copied,
            'missing': batch_missing,
            'total': len(batch_image_ids)
        })
        
        print(f"Batch {batch_num + 1}: {batch_copied} copied, {batch_missing} missing")
    
    print(f"\nOverall: Successfully copied {copied_count} images")
    print(f"Missing images: {missing_count}")
    
    return output_dir, batch_info

def check_rclone():
    """Check if rclone is installed and configured"""
    try:
        result = subprocess.run(['rclone', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ rclone is installed")
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        print("❌ rclone is not installed. Please install it first:")
        print("curl https://rclone.org/install.sh | sudo bash")
        return False
    
    # Check if Google Drive is configured
    try:
        result = subprocess.run(['rclone', 'listremotes'], capture_output=True, text=True)
        if 'gdrive:' in result.stdout:
            print("✓ Google Drive is configured")
            return True
        else:
            print("❌ Google Drive not configured. Please run: rclone config")
            return False
    except:
        print("❌ Error checking rclone configuration")
        return False

def upload_batches_to_gdrive(output_dir, batch_info, parent_folder_name="cops_ref_images"):
    """Upload each batch to Google Drive in separate folders"""
    print(f"\nUploading batches to Google Drive under '{parent_folder_name}'...")
    
    if not check_rclone():
        print("Please install and configure rclone, then run the upload manually.")
        return False
    
    success_count = 0
    
    for batch in batch_info:
        batch_num = batch['batch_num']
        batch_dir = batch['batch_dir']
        
        # Remote path: parent_folder/batch_X
        remote_path = f"gdrive:{parent_folder_name}/batch_{batch_num}"
        
        print(f"\nUploading batch {batch_num}...")
        print(f"Local: {batch_dir}")
        print(f"Remote: {remote_path}")
        
        try:
            # Use rclone sync to upload the batch
            result = subprocess.run([
                'rclone', 'sync', batch_dir, remote_path, '--progress'
            ], check=True, capture_output=False)
            
            print(f"✓ Batch {batch_num} uploaded successfully")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error uploading batch {batch_num}: {e}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Upload interrupted by user")
            break
    
    print(f"\nUpload summary: {success_count}/{len(batch_info)} batches uploaded successfully")
    return success_count == len(batch_info)

def create_batch_summary(output_dir, batch_info):
    """Create a summary file of the batches"""
    summary_path = os.path.join(output_dir, "batch_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Cops-Ref Images Batch Summary\n")
        f.write("=" * 40 + "\n\n")
        
        total_images = sum(batch['copied'] for batch in batch_info)
        f.write(f"Total images: {total_images}\n")
        f.write(f"Number of batches: {len(batch_info)}\n")
        f.write(f"Batch size: 100 (except last batch)\n\n")
        
        for batch in batch_info:
            f.write(f"Batch {batch['batch_num']}: {batch['copied']} images\n")
            f.write(f"  Directory: {batch['batch_dir']}\n")
            if batch['missing'] > 0:
                f.write(f"  Missing: {batch['missing']} images\n")
            f.write("\n")
    
    print(f"Batch summary saved to: {summary_path}")

def main():
    # Configuration - UPDATE THESE PATHS
    COPS_REF_JSON = "/Users/sammcmanagan/Desktop/Thesis/Model/data/cops_ref_test_sample_1000.json"  # Update this path
    LOCAL_GQA_DIR = "/Users/sammcmanagan/Downloads/images"    # Update this path
    PARENT_FOLDER_NAME = "cops_ref_images"              # Google Drive parent folder name
    BATCH_SIZE = 100                                    # Images per batch
    
    print("Starting Cops-Ref image extraction and upload pipeline...")
    print(f"Batch size: {BATCH_SIZE} images per batch")
    
    # Step 1: Extract image IDs
    try:
        image_ids = extract_cops_ref_image_ids(COPS_REF_JSON)
        print(f"Found {len(image_ids)} unique images in Cops-Ref")
    except FileNotFoundError:
        print(f"❌ Error: Could not find Cops-Ref JSON file at: {COPS_REF_JSON}")
        print("Please update the COPS_REF_JSON path in the script.")
        return
    except Exception as e:
        print(f"❌ Error reading Cops-Ref JSON: {e}")
        return
    
    # Step 2: Check if local GQA directory exists
    if not os.path.exists(LOCAL_GQA_DIR):
        print(f"❌ Error: Local GQA directory not found at: {LOCAL_GQA_DIR}")
        print("Please update the LOCAL_GQA_DIR path in the script.")
        return
    
    # Step 3: Copy images in batches
    try:
        output_dir, batch_info = copy_cops_ref_images_in_batches(
            image_ids, LOCAL_GQA_DIR, batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"❌ Error copying images: {e}")
        return
    
    # Step 4: Create batch summary
    create_batch_summary(output_dir, batch_info)
    
    # Step 5: Upload to Google Drive
    upload_success = upload_batches_to_gdrive(output_dir, batch_info, PARENT_FOLDER_NAME)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Local batched images are in: {output_dir}")
    print(f"Number of batches created: {len(batch_info)}")
    
    if upload_success:
        print(f"✓ All batches uploaded to Google Drive under '{PARENT_FOLDER_NAME}'")
    else:
        print("⚠️ Some uploads may have failed. Check the output above.")
        print("You can re-run the script to retry failed uploads.")
    
    print("="*60)

if __name__ == "__main__":
    main()