class Chunker(object):

    def __init__(self, n: int, n_chunks: int, requires_grad: bool, do_synchronize: bool = False):
        self.n = n
        self.n_chunks = n_chunks
        self.requires_grad = requires_grad
        self.do_synchronize = do_synchronize
        self.chunk_size = int(math.ceil(n / n_chunks))
        self.lower = [chunk * self.chunk_size for chunk in range(self.n_chunks)]
        self.upper = [(chunk + 1) * self.chunk_size for chunk in range(self.n_chunks)]
        self.upper[-1] = n

    def chunk(self,
              get_run: Callable[[int, int], Callable],
              *input_tensors: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        result = []
        for lower, upper in zip(self.lower, self.upper):
            if self.requires_grad:
                result.append(checkpoint(get_run(lower, upper), *input_tensors))
                if self.do_synchronize:
                    torch.cuda.synchronize()
            else:
                result.append(get_run(lower, upper)(*input_tensors))

        result = torch.cat(result)
        return result


def chunked_message_and_aggregate(
    adj_t: torch_sparse.SparseTensor,
    x: torch.Tensor,
    n_chunks: int = 8,
    aggregation_function: Optional[Callable[[torch_sparse.SparseTensor, torch.Tensor], torch.Tensor]] = None,
    **kwargs
) -> torch.Tensor:
    if aggregation_function is None:
        def aggregation_function(adj: torch_sparse.SparseTensor, x: torch.Tensor) -> torch.Tensor:
            return torch_sparse.matmul(adj, x, reduce='sum')

        if not adj_t.coo()[-1].requires_grad:
            return aggregation_function(adj_t, x)

    edge_weight, *rest = sparse_tensor_to_tuple(adj_t)

    def row_chunked_matmul(lower: int, upper: int):

        def row_chunked_matmul_run(edge_weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            adj = tuple_to_sparse_tensor(edge_weight, *rest)
            return aggregation_function(adj[lower:upper, :], x)
        return row_chunked_matmul_run

    chunker = Chunker(x.size(0), n_chunks, True)
    new_embeddings = chunker.chunk(
        lambda lower, upper: row_chunked_matmul(lower, upper),
        edge_weight, x
    )

    return new_embeddings