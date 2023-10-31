import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        # Your code goes here

    def update(self, val, n=1):
        # Your code goes here
        
def iou_score(output, target):
   # Your code goes here