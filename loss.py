class celoss:
    def __init__(self):
        weight = torch.tensor([4, 1], dtype=torch.float32).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=weight).cuda()

    def __call__(self, target, pre):
        # pre = torch.argmax(pre.detach().cpu(), dim=-1)
        # target = torch.argmax(target, dim=-1)
        return self.criterion(pre, target)