import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graphnn


class StudentModel(nn.Module):
    # GAT model
    def __init__(self,input_size,hidden_size,n_features):
        super(StudentModel, self).__init__()
        self.gat_1=graphnn.GATConv(input_size, hidden_size,4) #attention to get 1024 features, 4 attention heads
        self.gat_2=graphnn.GATConv(4*hidden_size, hidden_size,4) #attention to get 1024 features 4 attention heads
        self.gat_3=graphnn.GATConv(4*hidden_size,n_features,6,concat=False) # to get 121 features with 6 attention heads
        self.lin_1=nn.Linear(input_size,4*hidden_size)
        self.lin_2=nn.Linear(4*hidden_size,4*hidden_size)
        self.lin_3=nn.Linear(4*hidden_size,n_features)
        self.elu=nn.ELU()
        #Glorot init for each gat
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, graphnn.GATConv):
                if hasattr(module, 'lin'):
                    nn.init.xavier_uniform_(module.lin.weight)
                    if module.lin.bias is not None:
                        nn.init.zeros_(module.lin.bias)
                if hasattr(module, 'att'):
                    nn.init.xavier_uniform_(module.att)


    def forward(self, x, edge_index):
        x = F.elu(self.gat_1(x, edge_index) + self.lin_1(x))
        x = F.elu(self.gat_2(x, edge_index) + self.lin_2(x))
        x = self.gat_3(x, edge_index) + self.lin_3(x)
        return x


# # Initialize model
# model = StudentModel()

# ## Save the model
# torch.save(model.state_dict(), "model.pth")

# ### This is the part we will run in the inference to grade your model
# ## Load the model
# model = StudentModel()  # !  Important : No argument
# model.load_state_dict(torch.load("model.pth", weights_only=True))
# model.eval()
# print("Model loaded successfully")