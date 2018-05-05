import numpy as np
import json
from run_ga import main as try_params
import argparse
import os
import csv

def get_params():
    """
    Sample from the parameter space
    """
    pretrained = "glove.840B.300d" # fixed
    embed_dim = 300 # fixed
    batch_sz = 2 ** np.random.randint(0, 8) # powers of 2
    lr = np.random.choice([1, 3, 5, 7]) * np.random.choice([1e-4, 1e-5]) # categorical X log of 10
    x_dim = embed_dim + np.random.randint(0, 4) * 100 # higher than embedding
    hidden = x_dim + np.random.randint(0, 4) * 100 # higher than embedding and x_dim
    zy_dim = x_dim - np.random.randint(0, 4) * 50 # no constraint
    zg_dim = x_dim - np.random.randint(0, 4) * 50 # no constraint
    llm = np.random.randint(20, 100) # 20 to 100
    glm = np.random.randint(20, 100) # 20 to 100
    return pretrained, embed_dim, batch_sz, lr, x_dim, hidden, zy_dim, zg_dim, llm, glm

def construct_params():
    """
    Construct a set of hyper params and cache to a buffer
    """
    (pretrained, embed_dim, batch_sz, lr,
     x_dim, hidden, zy_dim, zg_dim, llm, glm) = get_params()
    config = {
        'pretrained': pretrained,
        'embed_dim': embed_dim,
        'batch_sz': batch_sz,
        'lr': lr,
        'x_dim': x_dim,
        'hidden': hidden,
        'zy_dim': zy_dim,
        'zg_dim': zg_dim,
        'llm': llm,
        'glm': glm
    }
    with open('../config/hypertune.json', 'w+') as f:
        json.dump(config, f)
    return config

def hypertune(args):
    output_path = args['output_path']
    signature = args['signature']
    run_id = args['run_id']
    output_path = os.path.join(output_path, "results_{}_{}.csv".format(signature, run_id))
    configs = construct_params()
    hyper_params = list(configs.keys())
    if not args['resume']:
        with open(output_path, 'w+') as f:
            logger = csv.writer(f)
            logger.writerow(hyper_params + ['valid_loss', 'valid_acc', 'test_f1_match', 'test_acc_match', 'test_f1_mismatch', 'test_acc_mismatch'])
    max_iter = args['epochs']  # maximum iterations/epochs per configuration
    eta = 3 # defines downsampling rate (default=3)
    logeta = lambda x: np.log(x)/np.log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max+1)):
        n = int(np.ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
        r = max_iter*eta**(-s) # initial number of iterations to run configurations for

        #### Begin Finite Horizon Successive Halving with (n,r)
        T = [construct_params() for _ in range(n)]
        for i in range(s+1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n*eta**(-i)
            args['epochs'] = int(r*eta**(i))
            val_losses = [try_params(args, t) for t in T]
            with open(output_path, 'a+') as f:
                logger = csv.writer(f)
                for idx, hyp in enumerate(T):
                    logger.writerow([hyp[k] for k in hyper_params] + list(val_losses[idx]))
            T = [T[i] for i in np.argsort(val_losses, axis=1)[0:int(n_i/eta)]]

if __name__ == "__main__":
    OPTIONS = argparse.ArgumentParser()
    OPTIONS.add_argument('--run_id', dest='run_id', type=int, default=1)
    OPTIONS.add_argument('--hypertune', dest='hypertune', action='store_true')
    OPTIONS.add_argument('--signature', dest='signature', type=str, default="") # e.g. {model}_{data}
    OPTIONS.add_argument('--model', dest='model', type=str, default="ga")
    OPTIONS.add_argument('--epochs', dest='epochs', type=int, default=500)
    OPTIONS.add_argument('--patience', dest='patience', type=int, default=20)
    OPTIONS.add_argument('--multinli_data_path', dest='multinli_data_path',
                         type=str, default='../../data/multinli_1.0/')
    OPTIONS.add_argument('--snli_data_path', dest='snli_data_path',
                         type=str, default='../../data/snli_1.0/')
    OPTIONS.add_argument('--data', dest='data', default='snli')
    OPTIONS.add_argument('--model_path', dest='model_path',
                         type=str, default='saved_model/')
    OPTIONS.add_argument('--output_path', dest='output_path',
                         type=str, default='results/')
    OPTIONS.add_argument('--gpu', dest='gpu', type=int, default=-1)
    OPTIONS.add_argument('--resume', dest='resume', action='store_true')

    ARGS = vars(OPTIONS.parse_args())
    hypertune(ARGS)