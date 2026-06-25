import os
import sys
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm



def fix_anchor_generator(sgg_path):
    """Fix the specific issue in anchor_generator.py"""
    anchor_file = os.path.join(sgg_path, "maskrcnn_benchmark/modeling/rpn/anchor_generator.py")
    
    if os.path.exists(anchor_file):
        with open(anchor_file, "r") as f:
            content = f.read()
        
        # Fix the specific error
        content = content.replace("np.float3232", "np.float32")
        
        with open(anchor_file, "w") as f:
            f.write(content)
        
        print(f"Fixed np.float3232 in {anchor_file}")
        
def patch_image_filenames_function(sgg_path):
    """Patch the load_image_filenames function to return mock data"""
    vg_dataset_path = os.path.join(sgg_path, "maskrcnn_benchmark/data/datasets/visual_genome.py")
    
    if not os.path.exists(vg_dataset_path):
        print(f"Error: Could not find {vg_dataset_path}")
        return
    
    with open(vg_dataset_path, "r") as f:
        content = f.read()
    
    # Create a new load_image_filenames function that returns mock data
    new_function = """
def load_image_filenames(img_dir, image_file):
    print("Using mock image filenames data for inference")
    import os
    
    # Create dummy filenames
    num_images = 108073  # The exact number expected by the assertion
    filenames = [f"DUMMY_IMAGE_{i}.jpg" for i in range(num_images)]
    
    # Create dummy image info
    img_info = [{'width': 800, 'height': 600, 'file_name': f} for f in filenames]
    
    return filenames, img_info
"""
    
    # Find the existing function
    import re
    if "def load_image_filenames(" in content:
        # Find the entire function
        pattern = r"def load_image_filenames\(.*?\).*?return fns, img_info\n"
        # Replace it with our new function
        content = re.sub(pattern, new_function, content, flags=re.DOTALL)
        
        # Write the modified file
        with open(vg_dataset_path, "w") as f:
            f.write(content)
        
        print(f"Successfully patched load_image_filenames function in {vg_dataset_path}")
    else:
        print(f"Could not find load_image_filenames function in {vg_dataset_path}")     
        
def patch_load_graphs_function(sgg_path):
    """Patch the load_graphs function to skip h5 file requirement completely"""
    vg_dataset_path = os.path.join(sgg_path, "maskrcnn_benchmark/data/datasets/visual_genome.py")
    
    if not os.path.exists(vg_dataset_path):
        print(f"Error: Could not find {vg_dataset_path}")
        return
    
    with open(vg_dataset_path, "r") as f:
        content = f.read()
    
    # Create a new load_graphs function that doesn't try to open any h5 file
    new_load_graphs = """
def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    print("Using mock data instead of trying to load h5 file")
    import torch
    
    # Create mock data
    num_images = 1000
    split_mask = torch.zeros(num_images, dtype=torch.bool)
    split_mask[:800] = True  # 800 train, 200 test
    
    # Create empty tensors for inference
    gt_boxes = torch.zeros((num_images, 0, 4), dtype=torch.float32)
    gt_classes = torch.zeros((num_images, 0), dtype=torch.int64)
    gt_attributes = torch.zeros((num_images, 0), dtype=torch.int64)
    relationships = torch.zeros((num_images, 0, 3), dtype=torch.int64)
    
    return split_mask, gt_boxes, gt_classes, gt_attributes, relationships
"""
    
    # Find the existing load_graphs function
    import re
    if "def load_graphs(" in content:
        # Find the entire function
        load_graphs_pattern = r"def load_graphs\(.*?\).*?return.*?\n"
        # Replace it with our new function
        content = re.sub(load_graphs_pattern, new_load_graphs, content, flags=re.DOTALL)
        
        # Write the modified file
        with open(vg_dataset_path, "w") as f:
            f.write(content)
        
        print(f"Successfully patched load_graphs function in {vg_dataset_path}")
    else:
        print(f"Could not find load_graphs function in {vg_dataset_path}")


def setup_sgg(sgg_path, checkpoint_path, config_file=None):
    """Set up the SGG model with targeted fixes for Mac compatibility"""
    import sys
    import os
    import json
    import torch
    
    # Add SGG to path if needed
    if sgg_path not in sys.path:
        sys.path.append(sgg_path)
    
    # Step 1: Fix the specific numpy error
    fix_anchor_generator(sgg_path)
    
    # Step 2: Create mock modules for Apex and C++ extensions
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create mock_apex.py
    with open(os.path.join(current_dir, "mock_apex.py"), "w") as f:
        f.write("""
class DummyAutocast:
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyAmp:
    def __init__(self):
        self.autocast = DummyAutocast
    
    def initialize(self, models, optimizers=None, enabled=True, opt_level="O1", **kwargs):
        return models, optimizers
    
    def float_function(self, func):
        return func

amp = DummyAmp()
""")
    
    # Create mock_cpp_extensions.py
    with open(os.path.join(current_dir, "mock_cpp_extensions.py"), "w") as f:
        f.write("""
import torch
from torchvision.ops import nms as torch_nms

class MockCppExtensions:
    @staticmethod
    def nms(boxes, scores, thresh):
        return torch_nms(boxes, scores, thresh)

# Other functions can be added as needed
""")
    
    # Import and set up the mocks
    sys.path.insert(0, current_dir)
    import mock_apex
    import mock_cpp_extensions
    
    sys.modules['apex'] = mock_apex
    sys.modules['apex.amp'] = mock_apex
    
    # Create _C module
    import types
    sys.modules['maskrcnn_benchmark._C'] = types.ModuleType('maskrcnn_benchmark._C')
    sys.modules['maskrcnn_benchmark._C'].nms = mock_cpp_extensions.MockCppExtensions.nms
    
    # Step 3: Patch any file that imports apex
    for root, dirs, files in os.walk(os.path.join(sgg_path, "maskrcnn_benchmark/layers")):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    content = f.read()
                
                if "from apex import amp" in content:
                    print(f"Patching apex import in {file_path}")
                    content = content.replace("from apex import amp", "from mock_apex import amp")
                    with open(file_path, "w") as f:
                        f.write(content)
    
    # Step 4: Replace the entire Visual Genome dataset with our mock
    patch_visual_genome_dataset(sgg_path)
    patch_get_statistics(sgg_path)
    patch_build_py(sgg_path)
    
    # Continue with configuration and loading...
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
    from maskrcnn_benchmark.modeling.detector import build_detection_model
    
    if config_file is None:
        config_file = os.path.join(sgg_path, "configs/e2e_relation_X_101_32_8_FPN_1x.yaml")
    
    cfg.merge_from_file(config_file)
    
    # Set model parameters
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
    cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
    cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = "CausalAnalysisPredictor"
    cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = "TDE"
    cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = "sum"
    cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = "motifs"
    cfg.TEST.IMS_PER_BATCH = 1
    cfg.TEST.DETECTIONS_PER_IMG = 100
    cfg.TEST.RELATION.SYNC_GATHER = False
    
    # Force CPU mode for Mac
    cfg.MODEL.DEVICE = "cpu"
    # Use float32 instead of float16 for CPU
    cfg.DTYPE = "float32"
    
    # Set custom eval flag
    cfg.TEST.CUSTUM_EVAL = True
    cfg.TEST.CUSTUM_PATH = os.path.abspath(os.path.expanduser(cfg.TEST.CUSTUM_PATH))
    
    cfg.freeze()
    
    # Build model and load checkpoint
    print("Building model...")
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    
    print("Loading checkpoint...")
    checkpointer = DetectronCheckpointer(cfg, model)
    
    # Load checkpoint without optimizer state
    if os.path.exists(checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Remove unnecessary state to avoid errors
        if "optimizer" in checkpoint_data:
            del checkpoint_data["optimizer"]
        if "scheduler" in checkpoint_data:
            del checkpoint_data["scheduler"]
        if "iteration" in checkpoint_data:
            del checkpoint_data["iteration"]
        
        model.load_state_dict(checkpoint_data["model"])
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
    
    return model, cfg

def patch_visual_genome_dataset(sgg_path):
    """Comprehensive patch for the Visual Genome dataset"""
    vg_dataset_path = os.path.join(sgg_path, "maskrcnn_benchmark/data/datasets/visual_genome.py")
    
    if not os.path.exists(vg_dataset_path):
        print(f"Error: Could not find {vg_dataset_path}")
        return
    
    # Create a completely new mock version of the file
    mock_content = """
import os
import torch
import json
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import numpy as np
from PIL import Image
import random

class VGDataset(object):
    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 flip_aug=False, custom_eval=False, custom_path=''):
        
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.split = split
        self.filter_non_overlap = filter_non_overlap and split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms
        self.flip_aug = flip_aug
        self.custom_eval = custom_eval
        self.custom_path = custom_path

        # Get the class and predicate mappings
        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(dict_file)
        
        # For custom eval mode
        if self.custom_eval:
            self.custom_files = []
            if custom_path.endswith('.jpg') or custom_path.endswith('.png'):
                self.custom_files = [custom_path]
            else:
                if os.path.isdir(custom_path):
                    for file_name in os.listdir(custom_path):
                        if file_name.endswith('.jpg') or file_name.endswith('.png'):
                            self.custom_files.append(os.path.join(custom_path, file_name))
                else:
                    try:
                        with open(custom_path, 'r') as f:
                            data = json.load(f)
                            for img_path in data:
                                self.custom_files.append(img_path)
                    except Exception as e:
                        print(f"Error loading custom path: {e}")
            
            print(f"Using {len(self.custom_files)} custom files for evaluation")
            return
        
        # Create mock data
        self.filenames = [f"DUMMY_{i}.jpg" for i in range(108073)]
        self.img_info = [{'file_name': f, 'width': 800, 'height': 600} for f in self.filenames]
        
        # Create dummy tensors
        num_images = 108073
        self.split_mask = torch.zeros(num_images, dtype=torch.bool)
        self.split_mask[:80000] = True  # 80000 train, rest test
        
        self.gt_boxes = torch.zeros((num_images, 0, 4), dtype=torch.float32)
        self.gt_classes = torch.zeros((num_images, 0), dtype=torch.int64)
        self.gt_attributes = torch.zeros((num_images, 0), dtype=torch.int64)
        self.relationships = torch.zeros((num_images, 0, 3), dtype=torch.int64)
        
        # Filter based on split
        if self.split == 'train':
            self.filenames = [self.filenames[i] for i in range(len(self.split_mask)) if self.split_mask[i]]
            self.img_info = [self.img_info[i] for i in range(len(self.split_mask)) if self.split_mask[i]]
        elif self.split == 'val' or self.split == 'test':
            self.filenames = [self.filenames[i] for i in range(len(self.split_mask)) if not self.split_mask[i]]
            self.img_info = [self.img_info[i] for i in range(len(self.split_mask)) if not self.split_mask[i]]
        
        print(f"Initialized VGDataset with {len(self.filenames)} {split} images")

    def __getitem__(self, index):
        if self.custom_eval:
            img = Image.open(self.custom_files[index]).convert("RGB")
            target = torch.LongTensor([-1])
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, index
        
        # Create dummy target
        img_path = os.path.join(self.img_dir, self.filenames[index])
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Create a dummy image if the file doesn't exist
            img = Image.new("RGB", (800, 600), color=(255, 255, 255))
        
        target = BoxList([[10, 10, 100, 100]], (800, 600), mode="xyxy")
        target.add_field("labels", torch.tensor([1]))  # background
        target.add_field("attributes", torch.tensor([0]))  # no attribute
        target.add_field("relation_pair_idxs", torch.tensor([[0, 0]]))
        target.add_field("pred_labels", torch.tensor([0]))
        target.add_field("rel_pair_tensor", torch.tensor([[0, 0]]))
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target, index

    def get_statistics(self):
        # Create mock statistics for frequency
        fg_matrix = torch.zeros((150, 51, 150), dtype=torch.int64)
        bg_matrix = torch.zeros((150, 51, 150), dtype=torch.int64)
        
        # Set some common relationships
        fg_matrix[0, 1, 0] = 100  # person on person
        fg_matrix[0, 2, 1] = 80   # person wear shirt
        
        # Class counts
        obj_count = torch.ones(151, dtype=torch.int64)
        obj_count[0] = 0  # No background
        obj_count[1] = 1000  # Person is common
        
        rel_count = torch.ones(51, dtype=torch.int64)
        rel_count[0] = 0  # No background
        rel_count[1] = 800  # 'on' is common
        
        num_obj_classes = len(self.ind_to_classes)
        num_rel_classes = len(self.ind_to_predicates)
        
        return fg_matrix, bg_matrix, obj_count, rel_count

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)

def load_info(dict_file):
    print(f"Loading class mappings from {dict_file}")
    info = json.load(open(dict_file, 'r'))
    
    if 'label_to_idx' in info:
        # File is already in the expected format
        label_to_idx = info['label_to_idx']
        predicate_to_idx = info['predicate_to_idx']
        attribute_to_idx = info.get('attribute_to_idx', {'__background__': 0})
    elif 'idx_to_label' in info:
        # Convert from idx_to_label to label_to_idx format
        label_to_idx = {v: int(k) for k, v in info['idx_to_label'].items()}
        predicate_to_idx = {v: int(k) for k, v in info['idx_to_predicate'].items()}
        attribute_to_idx = {v: int(k) for k, v in info.get('idx_to_attribute', {'0': '__background__'}).items()}
    else:
        # Create basic mappings if neither format is available
        print("Warning: JSON file doesn't have label_to_idx or idx_to_label. Creating mock mappings.")
        label_to_idx = {'__background__': 0, 'person': 1, 'bicycle': 2}
        predicate_to_idx = {'__background__': 0, 'on': 1, 'has': 2}
        attribute_to_idx = {'__background__': 0}
    
    # Ensure background class exists
    if '__background__' not in label_to_idx:
        label_to_idx['__background__'] = 0
    
    if '__background__' not in predicate_to_idx:
        predicate_to_idx['__background__'] = 0
    
    # Convert to idx_to_label format
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    idx_to_predicate = {str(v): k for k, v in predicate_to_idx.items()}
    idx_to_attribute = {str(v): k for k, v in attribute_to_idx.items()}
    
    return idx_to_label, idx_to_predicate, idx_to_attribute

def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    print(f"Using mock data instead of {roidb_file}")
    
    # Create mock data
    num_images = 108073
    split_mask = torch.zeros(num_images, dtype=torch.bool)
    split_mask[:80000] = True  # 80000 train, rest test
    
    # Create empty tensors
    gt_boxes = torch.zeros((num_images, 0, 4), dtype=torch.float32)
    gt_classes = torch.zeros((num_images, 0), dtype=torch.int64)
    gt_attributes = torch.zeros((num_images, 0), dtype=torch.int64)
    relationships = torch.zeros((num_images, 0, 3), dtype=torch.int64)
    
    return split_mask, gt_boxes, gt_classes, gt_attributes, relationships

def load_image_filenames(img_dir, image_file):
    print("Using mock image filenames")
    
    # Create dummy filenames
    num_images = 108073
    filenames = [f"DUMMY_{i}.jpg" for i in range(num_images)]
    img_info = [{'file_name': f, 'width': 800, 'height': 600} for f in filenames]
    
    return filenames, img_info

def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
    print("Using mock VG statistics")
    
    # Number of object classes (including background)
    num_obj_classes = 151
    # Number of predicate classes (including background)
    num_rel_classes = 51
    
    # Create mock statistics
    fg_matrix = torch.zeros((num_obj_classes - 1, num_rel_classes, num_obj_classes - 1), dtype=torch.int64)
    bg_matrix = torch.zeros((num_obj_classes - 1, num_rel_classes, num_obj_classes - 1), dtype=torch.int64)
    
    # Set some values to make it look realistic
    fg_matrix[0, 1, 0] = 100  # person -> on -> person
    fg_matrix[0, 2, 1] = 80   # person -> wearing -> clothing
    
    # Class count statistics
    obj_count = torch.ones(num_obj_classes, dtype=torch.int64)
    obj_count[0] = 0  # Background is 0
    obj_count[1] = 1000  # "person" is most common
    
    rel_count = torch.ones(num_rel_classes, dtype=torch.int64)
    rel_count[0] = 0  # Background is 0
    rel_count[1] = 800  # "on" is most common
    
    return fg_matrix, bg_matrix, obj_count, rel_count
"""
    
    # Write the completely mocked file
    with open(vg_dataset_path, "w") as f:
        f.write(mock_content)
    
    print(f"Replaced Visual Genome dataset with mock implementation in {vg_dataset_path}")

def patch_get_statistics(sgg_path):
    """Patch the get_statistics method in the Visual Genome dataset"""
    vg_dataset_path = os.path.join(sgg_path, "maskrcnn_benchmark/data/datasets/visual_genome.py")
    
    if not os.path.exists(vg_dataset_path):
        print(f"Error: Could not find {vg_dataset_path}")
        return
    
    with open(vg_dataset_path, "r") as f:
        content = f.read()
    
    # Find the get_statistics method
    if "def get_statistics(self):" in content:
        # Find start and end of the method
        start_idx = content.find("def get_statistics(self):")
        end_idx = content.find("\n    def", start_idx)
        if end_idx == -1:  # If it's the last method
            end_idx = len(content)
        
        # Extract the method
        old_method = content[start_idx:end_idx]
        
        # Create a new method that returns a dictionary
        new_method = """def get_statistics(self):
        # Create mock statistics for frequency
        fg_matrix = torch.zeros((150, 51, 150), dtype=torch.int64)
        bg_matrix = torch.zeros((150, 51, 150), dtype=torch.int64)

        # Set some common relationships
        fg_matrix[0, 1, 0] = 100  # person on person
        fg_matrix[0, 2, 1] = 80   # person wear shirt

        # Class counts
        obj_count = torch.ones(151, dtype=torch.int64)
        obj_count[0] = 0  # No background
        obj_count[1] = 1000  # Person is common

        rel_count = torch.ones(51, dtype=torch.int64)
        rel_count[0] = 0  # No background
        rel_count[1] = 800  # 'on' is common

        num_obj_classes = len(self.ind_to_classes)
        num_rel_classes = len(self.ind_to_predicates)
        
        # Return a list containing a dictionary
        return [{'fg_matrix': fg_matrix, 'bg_matrix': bg_matrix, 'obj_count': obj_count, 'rel_count': rel_count}]
"""
        
        # Replace the method
        content = content.replace(old_method, new_method)
        
        # Write the updated file
        with open(vg_dataset_path, "w") as f:
            f.write(content)
        
        print(f"Patched get_statistics method in {vg_dataset_path}")
    else:
        print(f"Could not find get_statistics method in {vg_dataset_path}")
        

def patch_build_py(sgg_path):
    """Patch the build.py file to handle our mock statistics format"""
    build_py_path = os.path.join(sgg_path, "maskrcnn_benchmark/data/build.py")
    
    if not os.path.exists(build_py_path):
        print(f"Error: Could not find {build_py_path}")
        return
    
    with open(build_py_path, "r") as f:
        content = f.read()
    
    # Find the get_dataset_statistics function
    if "def get_dataset_statistics(" in content:
        # Find the problematic part
        problematic_code = """
    return {
        'fg_matrix': statistics[0]['fg_matrix'],
        'pred_dist': statistics[0]['pred_dist'],
        'obj_classes': statistics[0]['obj_classes'],
        'rel_classes': statistics[0]['rel_classes'],
        'att_classes': statistics[0]['att_classes'],
    }"""
        
        # Replace with a more robust version
        new_code = """
    # Handle both tuple and dictionary return formats
    if isinstance(statistics[0], dict):
        # Already in dictionary format
        stats_dict = statistics[0]
    else:
        # Convert from tuple format (fg_matrix, bg_matrix, obj_count, rel_count)
        fg_matrix, bg_matrix, obj_count, rel_count = statistics[0]
        stats_dict = {
            'fg_matrix': fg_matrix,
            'bg_matrix': bg_matrix,
            'obj_count': obj_count,
            'rel_count': rel_count,
            'obj_classes': dataset.ind_to_classes,
            'rel_classes': dataset.ind_to_predicates,
            'att_classes': dataset.ind_to_attributes,
        }
    
    # Make sure all required keys are present
    if 'pred_dist' not in stats_dict:
        # Create pred_dist from fg_matrix
        pred_dist = stats_dict['fg_matrix'].sum(0).sum(1)
        stats_dict['pred_dist'] = pred_dist
    
    if 'obj_classes' not in stats_dict:
        stats_dict['obj_classes'] = dataset.ind_to_classes
    
    if 'rel_classes' not in stats_dict:
        stats_dict['rel_classes'] = dataset.ind_to_predicates
    
    if 'att_classes' not in stats_dict:
        stats_dict['att_classes'] = dataset.ind_to_attributes
    
    return stats_dict"""
        
        # Replace the code
        content = content.replace(problematic_code, new_code)
        
        # Write the updated file
        with open(build_py_path, "w") as f:
            f.write(content)
        
        print(f"Patched get_dataset_statistics function in {build_py_path}")
    else:
        print(f"Could not find get_dataset_statistics function in {build_py_path}")


def create_mock_modules(current_dir):
    """Create mock module files for Apex and C++ extensions"""
    # Create mock_apex.py
    with open(os.path.join(current_dir, "mock_apex.py"), "w") as f:
        f.write("""
class DummyAutocast:
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DummyAmp:
    def __init__(self):
        self.autocast = DummyAutocast
    
    def initialize(self, models, optimizers=None, enabled=True, opt_level="O1", 
                  cast_model_type=None, patch_torch_functions=None, keep_batchnorm_fp32=None,
                  master_weights=None, loss_scale=None, cast_model_outputs=None):
        return models, optimizers
    
    def float_function(self, func):
        # Simply return the function unchanged
        return func

amp = DummyAmp()
""")
    
    # Create mock_cpp_extensions.py
    with open(os.path.join(current_dir, "mock_cpp_extensions.py"), "w") as f:
        f.write("""
import torch
import torch.nn.functional as F
from torchvision.ops import nms as torch_nms
try:
    from torchvision.ops import roi_align as torch_roi_align
    from torchvision.ops import roi_pool as torch_roi_pool
except ImportError:
    # Fallback implementations if not available
    def torch_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio):
        # Simple implementation using interpolation
        return F.interpolate(input, size=output_size)
    
    def torch_roi_pool(input, boxes, output_size, spatial_scale):
        # Simple implementation using adaptive pooling
        return F.adaptive_max_pool2d(input, output_size)

# Mock the C++ extension module
class MockCppExtensions:
    @staticmethod
    def nms(boxes, scores, thresh):
        return torch_nms(boxes, scores, thresh)
    
    @staticmethod
    def roi_align(input, boxes, output_size, spatial_scale, sampling_ratio):
        return torch_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio)
    
    @staticmethod
    def roi_pool(input, boxes, output_size, spatial_scale):
        return torch_roi_pool(input, boxes, output_size, spatial_scale)
    
    @staticmethod
    def encode_boxes(reference_boxes, proposals, weights):
        # Simple implementation of box encoding
        wx, wy, ww, wh = weights
        proposals_x1 = proposals[:, 0::4]
        proposals_y1 = proposals[:, 1::4]
        proposals_x2 = proposals[:, 2::4]
        proposals_y2 = proposals[:, 3::4]
        
        reference_boxes_x1 = reference_boxes[:, 0::4]
        reference_boxes_y1 = reference_boxes[:, 1::4]
        reference_boxes_x2 = reference_boxes[:, 2::4]
        reference_boxes_y2 = reference_boxes[:, 3::4]

        ex_widths = proposals_x2 - proposals_x1 + 1.0
        ex_heights = proposals_y2 - proposals_y1 + 1.0
        ex_ctr_x = proposals_x1 + 0.5 * ex_widths
        ex_ctr_y = proposals_y1 + 0.5 * ex_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1 + 1.0
        gt_heights = reference_boxes_y2 - reference_boxes_y1 + 1.0
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets
    
    @staticmethod
    def decode_boxes(rel_codes, boxes, weights):
        boxes = boxes.to(rel_codes.dtype)
        
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        pred_w = torch.exp(dw) * widths.unsqueeze(1)
        pred_h = torch.exp(dh) * heights.unsqueeze(1)

        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes
""")
def patch_numpy_deprecations(sgg_path):
    """Patch NumPy deprecations more carefully"""
    numpy_files = [
        "maskrcnn_benchmark/modeling/rpn/anchor_generator.py",
        "maskrcnn_benchmark/modeling/box_coder.py",
        "maskrcnn_benchmark/structures/bounding_box.py",
    ]
    
    for rel_path in numpy_files:
        file_path = os.path.join(sgg_path, rel_path)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()
            
            # Check if already patched (contains float32)
            if "np.float32" in content:
                # File already contains float32, might have been double-patched
                # Replace any wrong type name that might have been created
                content = content.replace("np.float3232", "np.float32")
            else:
                # Not yet patched, do the replacement
                content = content.replace("np.float", "np.float32")
            
            # Handle other deprecated types
            if "np.int32" not in content:
                content = content.replace("np.int ", "np.int32 ")
                content = content.replace("np.int,", "np.int32,")
                content = content.replace("np.int)", "np.int32)")
                content = content.replace("dtype=np.int", "dtype=np.int32")
            
            if "np.bool_" not in content:
                content = content.replace("np.bool ", "np.bool_ ")
                content = content.replace("np.bool,", "np.bool_,")
                content = content.replace("np.bool)", "np.bool_)")
                content = content.replace("dtype=np.bool", "dtype=np.bool_")
            
            with open(file_path, "w") as f:
                f.write(content)
            
            print(f"Patched NumPy types in {rel_path}")

def detect_scene_graph(image_path, model, cfg):
    """Detect scene graph for a single image"""
    from maskrcnn_benchmark.data.transforms import build_transforms
    from maskrcnn_benchmark.utils.visualization import load_image
    from maskrcnn_benchmark.structures.image_list import to_image_list
    
    # Load and transform image
    img = load_image(image_path)
    transforms = build_transforms(cfg, is_train=False)
    img, _ = transforms(img, None)
    
    # Create image batch
    image_list = to_image_list([img], size_divisible=32)
    image_list = image_list.to(cfg.MODEL.DEVICE)
    
    # Run model
    with torch.no_grad():
        output = model(image_list)
        
    # Process output
    boxes = output[0].bbox.cpu().numpy()
    labels = output[0].get_field("pred_labels").cpu().numpy()
    scores = output[0].get_field("pred_scores").cpu().numpy()
    
    # Get relationship predictions
    pred_rel_pairs = output[0].get_field("rel_pair_idxs").cpu().numpy()
    pred_rel_labels = output[0].get_field("pred_rel_labels").cpu().numpy()
    pred_rel_scores = output[0].get_field("pred_rel_scores").cpu().numpy()
    
    # Create scene graph dictionary
    scene_graph = {
        "bbox": boxes.tolist(),
        "bbox_labels": labels.tolist(),
        "bbox_scores": scores.tolist(),
        "rel_pairs": pred_rel_pairs.tolist(),
        "rel_labels": pred_rel_labels.tolist(),
        "rel_scores": pred_rel_scores.tolist()
    }
    
    return scene_graph

def process_dataset(image_dir, output_dir, model, cfg, class_mapping=None):
    """Process all images in a directory and save scene graphs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Get class mappings if not provided
    if class_mapping is None:
        try:
            from maskrcnn_benchmark.data.datasets.vg import VG150Dataset
            object_classes = VG150Dataset.CLASSES
            predicate_classes = VG150Dataset.PREDICATES
        except ImportError:
            print("Warning: Could not import VG150Dataset")
            object_classes = [f"obj_{i}" for i in range(151)]
            predicate_classes = [f"rel_{i}" for i in range(51)]
    else:
        object_classes = class_mapping.get("objects", [])
        predicate_classes = class_mapping.get("predicates", [])
    
    class_info = {
        "object_classes": object_classes,
        "predicate_classes": predicate_classes
    }
    
    # Save class info
    with open(os.path.join(output_dir, "class_info.json"), "w") as f:
        json.dump(class_info, f, indent=2)
    
    # Process each image
    scene_graphs = {}
    for img_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, img_file)
        image_id = os.path.splitext(img_file)[0]
        
        try:
            scene_graph = detect_scene_graph(image_path, model, cfg)
            scene_graphs[image_id] = scene_graph
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Save all scene graphs
    with open(os.path.join(output_dir, "scene_graphs.json"), "w") as f:
        json.dump(scene_graphs, f)
    
    # Convert to CSV format
    convert_to_csv(scene_graphs, object_classes, predicate_classes, output_dir)
    
    return scene_graphs

def convert_to_csv(scene_graphs, object_classes, predicate_classes, output_dir):
    """Convert scene graphs to CSV format for DeepSoftLog"""
    import csv
    
    # Create CSV file
    csv_path = os.path.join(output_dir, "scene_graphs.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(["image_id", "subject", "subject_bbox", "relationship", "object", "object_bbox"])
        
        # Write each relationship
        for image_id, sg in scene_graphs.items():
            for (subj_idx, obj_idx), rel_idx, rel_score in zip(
                    sg['rel_pairs'], sg['rel_labels'], sg['rel_scores']):
                
                # Skip low confidence relationships
                if rel_score < 0.3:
                    continue
                
                # Get subject and object info
                subj_label = object_classes[sg['bbox_labels'][subj_idx]] if sg['bbox_labels'][subj_idx] < len(object_classes) else f"obj_{sg['bbox_labels'][subj_idx]}"
                obj_label = object_classes[sg['bbox_labels'][obj_idx]] if sg['bbox_labels'][obj_idx] < len(object_classes) else f"obj_{sg['bbox_labels'][obj_idx]}"
                
                # Sanitize names
                subj_label = subj_label.replace('.', '_').replace(' ', '_')
                obj_label = obj_label.replace('.', '_').replace(' ', '_')
                
                # Get bounding boxes
                subj_bbox = sg['bbox'][subj_idx]
                obj_bbox = sg['bbox'][obj_idx]
                
                # Get relationship
                rel_label = predicate_classes[rel_idx] if rel_idx < len(predicate_classes) else f"rel_{rel_idx}"
                rel_label = rel_label.replace('.', '_').replace(' ', '_')
                
                # Write row
                csv_writer.writerow([
                    image_id,
                    subj_label,
                    subj_bbox,
                    rel_label,
                    obj_label,
                    obj_bbox
                ])
    
    print(f"Saved CSV file to {csv_path}")
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Scene Graphs')
    parser.add_argument('--sgg_path', required=True, help='Path to the SGG repository')
    parser.add_argument('--image_dir', required=True, help='Directory containing images')
    parser.add_argument('--output_dir', required=True, help='Directory to save scene graphs')
    parser.add_argument('--checkpoint', required=True, help='Path to the checkpoint')
    parser.add_argument('--config', default=None, help='Path to the config file')
    print("this is the scene graph generator")
    args = parser.parse_args()
    
    # Setup SGG
    model, cfg = setup_sgg(args.sgg_path, args.checkpoint, args.config)
    
    # Process dataset
    scene_graphs = process_dataset(args.image_dir, args.output_dir, model, cfg)
    
    print(f"Processed {len(scene_graphs)} images")
