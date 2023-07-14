import torch

import pandas as pd

from torch import nn
from collections import OrderedDict

def get_weights_diff(original_model: OrderedDict, unlearn_model: OrderedDict) -> pd.DataFrame:
    diff_res = []
    criterion = nn.MSELoss()

    for key in original_model.keys():
        if any([x in key for x in ['bn', 'downsample']]):
            pass
        else:
            diff_res.append(
                [
                    key,
                    torch.sqrt(
                        criterion(
                            original_model[key].cpu(),
                            unlearn_model[key]
                        )
                    ).numpy()
                ]
            )
    diff_res = pd.DataFrame(
        diff_res, columns=['layer', 'diff_num']
    )
    diff_res['diff_num'] = diff_res['diff_num'].astype(float)

    return diff_res
