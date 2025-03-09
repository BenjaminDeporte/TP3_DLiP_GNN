import torch
import torch_geometric

# Duplicate of code in notebook

def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d | None = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    COMPLETE

    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None
        Is used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (remove it for the bonus)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    C, H, W = image.shape
    # (C, H, W) to (H*W, C)
    x = image.permute(1, 2, 0).reshape(-1, C)

    edge_index_list = []
    edge_attr_list = []
    
    # we create one "padding node" that will be used to link the border pixels to it.
    # this generic padding node has null features, so it will not be involved in the
    # kernel computation.
    data_padding_node = torch.zeros((1, C))
    id_padding_node = H * W  # linear index for the padding node, just after the last image pixel
    x = torch.cat([x, data_padding_node], dim=0)  # add the padding node to the features    

    # For each pixel at (i, j) (source node)
    for i in range(H):
        for j in range(W):
            src = i * W + j  # linear index for the source node
            # Loop over the 3x3
            for di in [-1, 0, 1]:
                ni = i + di
                for dj in [-1, 0, 1]:
                    nj = j + dj
                    # checking where the neighboring pixel is:
                    
                    # ... first case : within bounds
                    if 0 <= ni < H and 0 <= nj < W:
                        dst = ni * W + nj  # linear index for the neighbor
                    # ... second case : out of bounds, need padding
                    else:
                        dst = id_padding_node
                        
                    # add edge from neighbor to source
                    edge_index_list.append([dst, src])
                    # Save the relative offset as edge attribute,
                    # will be used to find the right kernel weight
                    edge_attr_list.append([di, dj])

    # tensors
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # shape: (2, num_edges)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)  # shape: (num_edges, 2)

    # Create and return the Data object
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d | None = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image.
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (remove it for the bonus)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."
        
    # channels
    C = data.size(1)
    # remove the padding node at the last position
    data_without_padding = data[:-1]
    # (H*W, C) to (H, W, C)
    image = data_without_padding.view(height, width, C)
    # Permute to (C, H, W) to get the original format
    image = image.permute(2, 0, 1)
    
    return image


class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # Don't forget to call the parent constructor with the correct aguments
        # we need aggr='add' as we will be adding the messages along the node feature dimension
        assert conv2d.bias is not None, "Conv2d layer should have bias=True"
        
        super(Conv2dMessagePassing, self).__init__(aggr='add')
        # storing the kernel weights and bias
        # the kernel weight is a tensor of shape (out_channels, in_channels, kernel_size[0], kernel_size[1])
        # ... cad ici (out_channels, node_feature_dimension, 3, 3)
        self.kernel_weights = conv2d.weight.data
        # the bias is a tensor of shape (out_channels), mais n'existe que si conv2d a été instancié avec bias=True
        self.bias = conv2d.bias.data  # shape (out_channels,)
        self.bias = self.bias / (conv2d.kernel_size[0] * conv2d.kernel_size[1])  # correct the bias for the sum

    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message through the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size E x out_channels ??)
        """
        
        # messages should be a tensor of size #edges x #channels_out
        messages = torch.zeros(x_j.size(0), self.kernel_weights.size(0))
        # loop over edges
        for e in range(x_j.size(0)):         
            # get the source node features
            x = x_j[e]  # size (in_channels,)
            # get the relative offset of the edge
            di, dj = edge_attr[e] # values in [-1, 0, 1]
            # compute indices to get the kernel weights
            id_i = di + 1
            id_i = id_i.int()
            id_j = dj + 1
            id_j = id_j.int()
            # compute the message
            # self.kernel_weights est de taille (out_channels, in_channels, kernel_size[0], kernel_size[1])
            # x est de taille (in_channels,)
            msg = torch.mul(self.kernel_weights[:, :, id_i, id_j], x) # output is of size (out_channels, in_channels)
            # msg = msg + self.bias  # add the bias, output is of size(out_channels, in_channels)
            msg = msg.sum(dim=1) # sum over the in_channels dimension, output is of size(out_channels,)
            msg = msg + self.bias  # add the bias, output is of size(out_channels, in_channels). NB : the bias has been divided by the number of messages !
            messages[e] = msg
            
        return messages