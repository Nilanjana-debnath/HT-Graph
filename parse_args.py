
import argparse
import sys


def init_parser():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('-d', '--data', type=str, default='wikipedia', help='Dataset name')
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')
    parser.add_argument('--root', type=str, default='.', help='Dataset root')
    parser.add_argument('--dim', type=int, default=None, help='Feature dimension')
    parser.add_argument('--no_feat_buffer', action='store_true', 
                        help='Do not pre-load data into GPU')
    # Model
    parser.add_argument('--n_layers', type=int, default=1, help='Number of graph layers')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Number of temporal neighbors')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of graph heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--strategy', type=str, default="recent_edges",
                        choices=["recent_nodes", "recent_edges", "uniform"],
                        help='Sampling strategy for temporal aggregation')
    parser.add_argument('--msg_src', type=str, default="left", choices=["left", "right"],
                        help='message source')
    parser.add_argument('--upd_src', type=str, default="right", choices=["left", "right"],
                        help='update source')
    parser.add_argument('--upd_fn', type=str, default="gru", choices=["merge", "gru"],
                        help='memory update function')
    parser.add_argument('--tsfm_fn', type=str, default="id", choices=["id", "linear", "mlp"],
                        help='message transform function')
    parser.add_argument('--hit_type', type=str, default="bin",
                        choices=["vec", "bin", "count", "none"], help='Hit type')
  
    # neighbourhood-related hyper-parameters
    parser.add_argument('--n_degree', nargs='*', default=['32', '16'],
              help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
    parser.add_argument('--n_hop', type=int, default=2, help='number of hops the N-cache is used')
    parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference with time closeness, default to 0 which is uniform sampling')
    parser.add_argument('--pos_dim', type=int, default=16, help='dimension of the positional embedding')
    parser.add_argument('--self_dim', type=int, default=72, help='dimension of the self representation')
    parser.add_argument('--ngh_feat_dim', type=int, default=4, help='dimension of the neighborhood representation')
    parser.add_argument('--linear_out', action='store_true', default=False, help="whether to linearly project each node's ")

    parser.add_argument('--attn_n_head', type=int, default=2, help='number of heads used in tree-shaped attention layer, we only use the default here')
    parser.add_argument('--time_dim', type=int, default=1, help='dimension of the time embedding')
    
    # -- restarter
    parser.add_argument('--mutual_coef', type=float, default=1.0, help='Mutual loss coef')
    parser.add_argument('--restart_prob', type=float, default=0.01, help='restart probability (train)')
    parser.add_argument('--restarter_type', type=str, default="seq",
                        choices=["seq", "static"], help='Restarter type')
    parser.add_argument('--hist_len', type=int, default=40, help='Length of history')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warmup steps')
    
    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--replace_prob', type=float, default=0.9, help='probability for storing new neighbors to N-cache replacing old ones')
    parser.add_argument('--tolerance', type=float, default=1e-3,
              help='toleratd margainal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    # parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    # parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
    parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')
    parser.add_argument('--run', type=int, default=2, help='number of model runs')
    
    
    try:
      args = parser.parse_args()
    except:
      parser.print_help()
      sys.exit(0)
    return args, sys.argv
