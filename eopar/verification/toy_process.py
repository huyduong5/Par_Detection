"""Subprocess fixture exercising production main/resume with a tiny CPU model."""
import sys
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import Dataset
from aq_visibility.losses.climpar_loss import CLIMPPARLoss
from types import SimpleNamespace
import eopar.train.run as run
from eopar.train.runtime import source_hashes, environment_identity


class Data(Dataset):
    attributes=['a','b']
    epoch=0
    def __init__(self,split): self.split=split
    def __len__(self): return 7
    def __getitem__(self,i):
        # Epoch-dependent deterministic pair and global RNG exercise restoration.
        x=torch.tensor([i/10., 1.])+torch.rand(2)*.001
        return dict(img_clean=x,img_occ=x*.9,domain_token=torch.zeros(1),
                    labels=torch.tensor([i%2,(i+1)%2]).float(),imgname=str(i),
                    vis_target=torch.ones(2)*.85,vis_valid=torch.ones(2),
                    intact_mask=torch.ones(2),affected_mask=torch.zeros(2),rank_weights=torch.zeros(2))


class Model(nn.Module):
    def __init__(self): super().__init__(); self.linear=nn.Linear(2,2)
    def forward(self,x,t,a):
        z=self.linear(x)
        return dict(attr_logits=z,observability_logits=None,legacy_visibility=z.sigmoid())


def build(c,out,initialize=True):
    crit=CLIMPPARLoss(SimpleNamespace(loss_type='weighted_bce_asl',lambda_v=.5,lambda_c=.2,
        use_visibility=True,consistency_type='sg_bce',global_contrastive_weight=0.,finegrained_contrastive_weight=0.))
    return Model(),crit


if __name__=='__main__':
    sys.argv.append('--verification-run')
    with patch.object(run,'build',build), patch.object(run,'make_dataset',lambda c,s:Data(s)), \
         patch.object(run,'manifest',lambda c,m:dict(config=c,source_hashes=source_hashes(),environment=environment_identity())), \
         patch.object(torch.cuda,'is_available',lambda:False):
        run.main()
