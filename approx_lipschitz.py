import torch


class LipschitzCalculator():
    def __init__(self, net, eps, step_size, num_steps):
        super(LipschitzCalculator, self).__init__()
        self.net = net
        self.step_size = step_size
        self.num_steps = num_steps
        self.eps = eps

    def __call__(self, inputs):
        requires_grads = [x.requires_grad for x in self.net.parameters()]
        self.net.requires_grad_(False)

        x = inputs.detach()
        targets = self.net(x)
        ans = torch.zeros_like(targets)

        for label in range(targets.size(1)):
            init_noise = torch.zeros_like(x).normal_(0, self.eps / 4)
            x = x + torch.clamp(init_noise, -self.eps / 2, self.eps / 2)

            for i in range(self.num_steps):
                x.requires_grad_()
                logits = self.net(x)
                loss = (logits[:, label] - targets[:, label]).abs().sum()
                loss.backward()
                x = torch.add(x.detach(), torch.sign(x.grad.detach()), alpha=self.step_size)
                x = torch.min(torch.max(x, inputs - self.eps), inputs + self.eps)

            with torch.no_grad():
                logits = self.net(x)
                ans[:, label] = logits[:, label]

        diff = torch.norm(ans - targets, dim=1, p=float('inf')) / self.eps
        for p, r in zip(self.net.parameters(), requires_grads):
            p.requires_grad_(r)
        return diff


def cal_Lipschitz(net, data_loader, eps=1/255, num_steps=20, restart=5, gpu=0):
    import os
    file_name = ''
    for j in range(100):
        file_name = 'lipschitz%d.txt' % j
        if not os.path.exists(file_name):
            break
    fp = open(file_name, 'w')

    cal = LipschitzCalculator(net, eps, eps / 4, num_steps)
    net.eval()
    ans_list = []
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.cuda(gpu)
        ans = torch.zeros(inputs.size(0), device=inputs.device)
        print(batch_idx)
        for _ in range(restart):
            lipschitz = cal(inputs)
            ans = torch.maximum(ans, lipschitz)
        ans_list += list(ans.cpu().numpy())

    fp.writelines(['%.4f\n' % x for x in ans_list])
    fp.close()
