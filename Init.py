import torch
import torch.nn as nn
import enum

class Methods(enum.Enum):
    Xavier_Normal = 0
    Xavier_Uniform = 1
    Kaming_Normal = 2
    Kaming_Uniform = 3
    Identity = 4
    Atkinson = 5
    Atkinson_General = 6

class InitMethod():
    def __init__(self, method):
        self.name = method.name
        self.id = method
    
    def init_network(self):
        pass

class XavierNormalInit(InitMethod):
    def __init__(self):
        InitMethod.__init__(self, Methods.Xavier_Normal)
    
    def init_network(self, net):
        for mod in net.modules():
            with torch.no_grad():
                try:
                    nn.init.xavier_normal_(mod.weight)
                    mod.bias.fill_(0.0)
                except:
                    continue

class XavierUniformInit(InitMethod):
    def __init__(self):
        InitMethod.__init__(self, Methods.Xavier_Uniform)
    
    def init_network(self, net):
        for mod in net.modules():
            with torch.no_grad():
                try:
                    nn.init.xavier_normal_(mod.weight)
                    mod.bias.fill_(0.0)
                except:
                    continue

class KamingNormalInit(InitMethod):
    def __init__(self):
        InitMethod.__init__(self, Methods.Kaming_Normal)
    
    def init_network(self, net):
        for mod in net.modules():
            with torch.no_grad():
                try:
                    nn.init.kaiming_normal_(mod.weight)
                    mod.bias.fill_(0.0)
                except:
                    continue

class KamingUniformInit(InitMethod):
    def __init__(self):
        InitMethod.__init__(self, Methods.Kaming_Uniform)
    
    def init_network(self, net):
        for mod in net.modules():
            with torch.no_grad():
                try:
                    nn.init.kaiming_uniform_(mod.weight)
                    mod.bias.fill_(0.0)
                except:
                    continue

class IdentityInit(InitMethod):
    def __init__(self):
        InitMethod.__init__(self, Methods.Identity)
    
    def init_network(self, net):
        for mod in net.modules():
            with torch.no_grad():
                try:
                    nn.init.identity_(mod.weight)
                    mod.bias.fill_(0.0)
                except:
                    continue

class AtkinsonInit(InitMethod):
    def __init__(self):
        InitMethod.__init__(self, Methods.Atkinson)
    
    def init_network(self, net):
        for mod in net.modules():
            try:
                mean = mod.init_data.mean
                var = mod.init_data.var

                fan_in = mod.weight.size(1)
                receptive_field_size = 1
                if mod.weight.dim() > 2:
                    receptive_field_size = mod.weight[0][0].numel()
                
                fan_in = fan_in * receptive_field_size
                with torch.no_grad():
                    # correction factor
                    co = mean**2+var
                    mod.weight.normal_(0.0, (1/(co*fan_in))**(1/2))
                    mod.bias.fill_(0.0)
            except:
                continue

class AtkinsonGeneralInit(InitMethod):
    def __init__(self, training_data, rounds=5):
        InitMethod.__init__(self, Methods.Atkinson_General)
        self.rounds = rounds
        self.training_data = training_data
    
    def moment_loss(self, moment, data):
        goal = 0
        if moment == 2:
            goal = 1
        if moment == 3:
            goal = 0
        if moment == 4:
            goal = 3

        d = data**moment
        means = torch.mean(d, axis=0)
        goal = torch.full_like(means, goal)
        loss = torch.sum(torch.abs(means-goal))

        return loss

    def init_network(self, net):
        # Pre-Train the network
        pre_optim = torch.optim.Adam(net.parameters())
        for _ in range(self.rounds):
            for data, _ in self.training_data:
                data = data.cuda()

                pre_optim.zero_grad()
                net(data)

                # Train all 4 moments
                loss_m = self.moment_loss(1, net.pre_activations())
                loss_s = self.moment_loss(2, net.pre_activations())
                loss_w = self.moment_loss(3, net.pre_activations())
                loss_k = self.moment_loss(4, net.pre_activations())
                loss = loss_m+loss_s+loss_w+loss_k

                net.reset_pre_activations()

                loss.backward()
                pre_optim.setp()