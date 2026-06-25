# ============================================================================
# PERMANENT VISUAL GENOME SETUP (One-time download to Google Drive)
# ============================================================================

def setup_permanent_vg_dataset():
    """Set up VG dataset in Google Drive once, then reuse it"""
    
    print("💾 PERMANENT VG DATASET SETUP")
    print("=" * 35)
    print("🎯 Goal: Download once to Google Drive, reuse forever")
    print("📁 Location: /content/drive/MyDrive/VG_Dataset/")
    print("💾 Size: ~15GB (one-time download)")
    print("⏱️ Setup time: 1-2 hours (once)")
    print("🔄 Future usage: <5 minutes (just copy from Drive)")
    print()
    from google.colab import drive
    import os
    import subprocess
    
    drive.mount('/content/drive')
    
    # Check if already downloaded to Drive
    drive_vg_dir = "/content/drive/MyDrive/VG_Dataset"
    drive_images_dir = f"{drive_vg_dir}/VG_100K"
    
    if os.path.exists(drive_images_dir):
        images = [f for f in os.listdir(drive_images_dir) if f.endswith('.jpg')]
        print(f"📊 Found {len(images)} images already in Google Drive")
        
        if len(images) >= 100000:
            print("✅ Full dataset already available in Google Drive!")
            return True
        elif len(images) > 1000:
            print("✅ Partial dataset available in Google Drive!")
            print("💡 This should be sufficient for training")
            return True
        else:
            print("⚠️ Very few images in Drive - need to download more")
    else:
        print("📁 No VG dataset found in Google Drive")
    
    print("\n🔄 Setting up permanent dataset...")
    
    # Create Drive directory
    os.makedirs(drive_images_dir, exist_ok=True)
    
    # Download strategy: Download directly to Google Drive
    print("📥 Downloading VG images to Google Drive...")
    print("⚠️ This will take 1-2 hours but only needs to be done ONCE")
    print()
    
    response = input("🤔 Proceed with one-time download? (y/n): ")
    
    if response.lower() == 'y':
        try:
            print("📥 Downloading Visual Genome images...")
            
            # Download Part 1 
            print("📦 Part 1: Downloading images.zip...")
            result1 = subprocess.run([
                "wget", "-O", f"{drive_vg_dir}/images.zip",
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
            ], timeout=7200)  # 2 hour timeout
            
            if result1.returncode == 0:
                print("📦 Extracting Part 1...")
                subprocess.run(["unzip", f"{drive_vg_dir}/images.zip", "-d", drive_vg_dir])
                subprocess.run(["rm", f"{drive_vg_dir}/images.zip"])
                print("✅ Part 1 complete")
            
            # Download Part 2
            print("📦 Part 2: Downloading images2.zip...")
            result2 = subprocess.run([
                "wget", "-O", f"{drive_vg_dir}/images2.zip", 
                "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
            ], timeout=7200)
            
            if result2.returncode == 0:
                print("📦 Extracting Part 2...")
                subprocess.run(["unzip", f"{drive_vg_dir}/images2.zip", "-d", drive_vg_dir])
                subprocess.run(["rm", f"{drive_vg_dir}/images2.zip"])
                print("✅ Part 2 complete")
            
            # Verify
            final_images = [f for f in os.listdir(drive_images_dir) if f.endswith('.jpg')]
            print(f"📊 Final count: {len(final_images)} images")
            
            if len(final_images) >= 100000:
                print("🎉 SUCCESS! VG dataset permanently stored in Google Drive")
                return True
            else:
                print("⚠️ Partial success - some images missing but should work")
                return True
                
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    else:
        print("⏸️ Download skipped")
        return False

def use_permanent_vg_dataset():
    """Use the permanently stored VG dataset from Google Drive"""
    
    print("📂 USING PERMANENT VG DATASET")
    print("=" * 35)
    
    import os
    import shutil
    
    # Check if dataset exists in Drive
    drive_images_dir = "/content/drive/MyDrive/VG_Dataset/VG_100K"
    
    if not os.path.exists(drive_images_dir):
        print("❌ No permanent dataset found in Google Drive")
        print("💡 Run setup_permanent_vg_dataset() first")
        return False
    
    images = [f for f in os.listdir(drive_images_dir) if f.endswith('.jpg')]
    print(f"📊 Found {len(images)} images in Google Drive")
    
    # Create local dataset directory
    local_images_dir = "/content/Scene/datasets/vg/VG_100K"
    os.makedirs(local_images_dir, exist_ok=True)
    
    # Copy from Drive to local (much faster than downloading)
    print("📂 Copying images from Google Drive to local storage...")
    
    try:
        # Use cp command for faster copying
        result = subprocess.run([
            "cp", "-r", f"{drive_images_dir}/*", local_images_dir
        ], shell=True)
        
        if result.returncode == 0:
            local_images = [f for f in os.listdir(local_images_dir) if f.endswith('.jpg')]
            print(f"✅ Copied {len(local_images)} images to local storage")
            print("🚀 Dataset ready for training!")
            return True
        else:
            print("❌ Copy failed")
            return False
            
    except Exception as e:
        print(f"❌ Copy error: {e}")
        return False

print("💾 Permanent Dataset Setup")
print("=" * 25)
print("💡 One-time setup, permanent reuse:")
print("   1. setup_permanent_vg_dataset()  # Download once to Drive")
print("   2. use_permanent_vg_dataset()    # Copy from Drive (fast)")
print()
print("⏱️ First time: 1-2 hours")
print("🔄 Every other time: <5 minutes")
setup_permanent_vg_dataset()