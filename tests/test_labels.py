import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from v1_core import read_maps, remap_mask, single_collate
import torch

maps=read_maps(str(Path(__file__).resolve().parents[1]/'configs'/'label_maps.json'))
raw=np.arange(8,dtype=np.uint8).reshape(2,4)
expected=np.array([[255,0,1,2],[3,4,5,6]],dtype=np.uint8)
assert np.array_equal(remap_mask(raw,'LoveDA',maps),expected)
import copy

fake_maps = copy.deepcopy(maps)
fake_maps["Potsdam"]["verified"] = False

try:
    remap_mask(raw, "Potsdam", fake_maps)
    raise AssertionError("Unverified map should fail")
except ValueError:
    pass
try:
    remap_mask(np.array([[8]],dtype=np.uint8),'LoveDA',maps)
    raise AssertionError('Unknown ID should fail')
except ValueError:
    pass
batch=single_collate([{'input_ids':torch.ones(15,dtype=torch.long),
                       'pixel_values':torch.randn(324,1176),
                       'image_grid_thw':torch.tensor([[1,36,36]]),
                       'target_mask':torch.zeros((512,512),dtype=torch.long),
                       'dataset_name':'LoveDA'}])
assert batch['input_ids'].shape==(1,15)
assert batch['pixel_values'].shape==(324,1176)
assert batch['image_grid_thw'].shape==(1,3)
assert batch['target_mask'].shape==(1,512,512)
print('PASS: LoveDA mapping, unverified refusal, invalid ID, batch shapes')
