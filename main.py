from model.train import train_with_gin
from eval.evaluate import eval_with_gin
from data.preprocess.utils import build_few_label_dataset
import gin
import argparse
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self supervised pretraining lib for ICU data')
    parser.add_argument('-m', '--method', default='train_eval', dest="method", required=False, type=str,
                        help='Method to call between : eval, train_eval, fl_eval.')
    parser.add_argument('-tc', '--trainconfig', default=None, dest="train_config", required=False, type=str,
                        help="Path to the gin train/encoder config file.")
    parser.add_argument('-ec', '--evalconfig', default=None, dest="eval_config", nargs='+', required=False, type=str,
                        help="Path to the gin eval/head config file.")
    parser.add_argument('-l', '--logdir', dest="logdir", required=False, type=str,
                        help="Path to the log directory ")
    parser.add_argument('-o', '--overwrite', default=False, dest="overwrite", required=False, type=bool,
                        help="Boolean to overwrite previous model in logdir")
    parser.add_argument('-sd', '--seed', default=1111, dest="seed", required=False,nargs='+', type=int, 
                        help="Random seed at training and evaluation, default : 1111")
    parser.add_argument('-ss', '--split_seed', default=1234, dest="split_seed", required=False,nargs='+', type=int,
                       help="Splittin seed in the case of few labels training, default : 1111")
    parser.add_argument('-s', '--strategy', default=None, dest="strategy", action='store', required=False, type=str,
                        help="Distribution strategy. We advise against using it. Can only bet set to 'mirrored' ")
    parser.add_argument('-w', '--window', default=None, dest="window", required=False, nargs='+', type=int,
                        help="Window for the unsupervised neighbor definition")
    parser.add_argument('-a', '--alpha', default=None, dest="alpha", required=False, nargs='+', type=float,
                        help="Trade-off parameter for the NCL objective")
    parser.add_argument('-rs', '--random_search', default=False, dest="rs", required=False, type=bool,
                        help="Boolean to set whether or not we are doing a random search. \
                        If True, parameters are sampled randomly among the lists provided.")
    parser.add_argument('-d', '--debug', default=False, dest="debug", required=False, type=bool,
                        help=" Boolean parameter to debug tensorflow code that switches to eager execution")
    parser.add_argument('-en', '--evalname', default=['eval'], dest="eval_name", nargs='+', required=False, type=str,
                        help="Name for the directories to store each type of evaluation. Has to be the same length as --evalconfig.")
    parser.add_argument('-p', '--percentage', default=100, dest="percentage", required=False, type=float,
                        help="Percentage of labels to consider when doing few labels setting. ")
    parser.add_argument('-mom', '--momentum', default=None, dest="momentum", nargs='+', required=False, type=float,
                        help=" Momentum paramater.")
    parser.add_argument('-tau', '--temperature', default=None, dest="temperature", nargs='+', required=False, type=float,
                        help="Temperature parameter.")
    parser.add_argument('-sdo', '--spatial_dropout', default=None, dest="sdo", nargs='+', required=False, type=float,
                        help="Rate for spatial dropout.")
    parser.add_argument('-tcos', '--time_cutout_size', default=None, dest="tcos", nargs='+', required=False, type=int,
                        help="Size of the history cutouts.")
    parser.add_argument('-tcop', '--time_cutout_proba', default=None, dest="tcop", nargs='+', required=False, type=float,
                        help="Probability of applying history cutout.")
    parser.add_argument('-gb', '--gaussian_bias', default=None, dest="gb", nargs='+', required=False, type=float,
                        help="Standard deviation of the Gaussian noise.")
    parser.add_argument('-tch', '--temporal_crop_history', default=None, dest="tch", nargs='+', required=False, type=float,
                        help="Minimum proportion of history to preserve")

    args = parser.parse_args()
    gin_bindings = [] 
    log_dir = args.logdir
    
    # We update name and gin_bindings according to parsed arguments
    if args.window :
        if args.rs:
            window = args.window[np.random.randint(len(args.window))]
        else:
            window = args.window[0]
        gin_bindings += ['WINDOW = '+str(window)]
        log_dir = log_dir.rstrip('/') + '_w_' + str(window)
    if args.alpha :
        if args.rs:
            alpha = args.alpha[np.random.randint(len(args.alpha))]
        else:
            alpha = args.alpha[0]
        gin_bindings += ['ALPHA  = ' +str(alpha)]
        log_dir = log_dir.rstrip('/') + '_a_' + str(alpha)
    if args.temperature:
        if args.rs:
            temp = args.temperature[np.random.randint(len(args.temperature))]
        else:
            temp = args.temperature[0]
        gin_bindings += ['TEMPERATURE  = ' +str(temp)]
        log_dir = log_dir.rstrip('/') + '_t_' + str(temp)
    if args.momentum:
        if args.rs:
            momentum = args.momentum[np.random.randint(len(args.momentum))]
        else:
            momentum = args.momentum[0]
        gin_bindings += ['MOMENTUM  = ' +str(momentum)]
        log_dir = log_dir.rstrip('/') + '_m_' + str(momentum)
    if args.sdo:
        if args.rs:
            sdo = args.sdo[np.random.randint(len(args.sdo))]
        else:
            sdo = args.sdo[0]
        gin_bindings += ['SDO  = ' +str(sdo)]
        log_dir = log_dir.rstrip('/') + '_sdo_' + str(sdo)
    if args.tcos:
        if args.rs:
            tcos = args.tcos[np.random.randint(len(args.tcos))]
        else:
            tcos = args.tcos[0]
        gin_bindings += ['TCO_SIZE  = ' +str(tcos)]
        log_dir = log_dir.rstrip('/') + '_tcos_' + str(tcos)
    if args.tcop:
        if args.rs:
            tcop = args.tcop[np.random.randint(len(args.tcop))]
        else:
            tcop = args.tcop[0]
        gin_bindings += ['TCO_PROBA  = ' +str(tcop)]
        log_dir = log_dir.rstrip('/') + '_tcos_' + str(tcop)
    if args.tch:
        if args.rs:
            tch = args.tch[np.random.randint(len(args.tch))]
        else:
            tch = args.tch[0]
        gin_bindings += ['TC  = ' +str(tch)]
        log_dir = log_dir.rstrip('/') + '_tc_' + str(tch)
    if args.gb:
        if args.rs:
            gb = args.gb[np.random.randint(len(args.gb))]
        else:
            gb = args.gb[0]
        gin_bindings += ['GB  = ' +str(gb)]
        log_dir = log_dir.rstrip('/') + '_gb_' + str(gb)
    if args.debug:
        import tensorflow as tf
        tf.config.experimental_run_functions_eagerly(True)
    if args.rs:
        args.seed = args.seed[np.random.randint(len(args.seed))]
        args.overwrite = False
    
    
    # Method used for frozen eval on pre-trained encoder or End-to-end training.
    if args.method == 'eval':
        if not isinstance(args.seed, list):
            seeds = [args.seed]
        else:
            seeds = args.seed
        for seed in seeds:
            if not isinstance(args.eval_config, list):
                eval_configs = [args.eval_config]
                if args.eval_name:
                    eval_names = [args.eval_name]
                else:
                    eval_names = ['eval']
            else:
                eval_configs = args.eval_config
                eval_names = args.eval_name
                assert len(eval_configs) == len(eval_names)
        
            log_dir_seed = os.path.join(log_dir,str(seed))
            train_dir = os.path.join(log_dir_seed, "train")
            gin_bindings += ['load_representation.model_dir = '+"'"+str(train_dir)+"'"]
            for i,ev in enumerate(eval_configs):
                eval_dir = os.path.join(log_dir_seed, eval_names[i])
                eval_gin_files = [args.train_config, ev]
                eval_with_gin(gin_config_files=eval_gin_files, strategy_name=args.strategy, model_dir=eval_dir,
                              overwrite=args.overwrite, seed=seed, gin_bindings=gin_bindings)
    
    # Method used for train and evaluating sequentially representations.
    elif args.method == 'train_eval':
        if not isinstance(args.seed, list):
            seeds = [args.seed]
        else:
            seeds = args.seed
        for seed in seeds:
            log_dir_seed = os.path.join(log_dir,str(seed))
            train_dir = os.path.join(log_dir_seed, "train")
            if not isinstance(args.eval_config, list):
                eval_configs = [args.eval_config]
                if args.eval_name:
                    eval_names = [args.eval_name]
                else:
                    eval_names = ['eval']
            else:
                eval_configs = args.eval_config
                eval_names = args.eval_name
                assert len(eval_configs) == len(eval_names)
            if args.overwrite or (not os.path.isdir(train_dir)):
                train_with_gin(gin_config_files=args.train_config, gin_bindings=gin_bindings, strategy_name=args.strategy, model_dir=train_dir,
                                overwrite=args.overwrite, seed=seed)
            else:
                print('Already trained, going to eval')
            gin_bindings += ['load_representation.model_dir = '+"'"+str(train_dir)+"'"]
            for i,ev in enumerate(eval_configs):
                eval_dir = os.path.join(log_dir_seed, eval_names[i])
                eval_gin_files = [args.train_config, ev]
                eval_with_gin(gin_config_files=eval_gin_files, strategy_name=args.strategy, model_dir=eval_dir,
                              overwrite=args.overwrite, seed=seed, gin_bindings=gin_bindings)
    
    
    # Method to evaluate representation on fractions of the labeled data.
    elif args.method == 'fl_eval':
        gin.parse_config_files_and_bindings([args.train_config], [])
        data_dir = gin.query_parameter('ICU_loader_semi_temporal.data_path')
        task  = gin.query_parameter('ICU_loader_semi_temporal.task')
        scaled_bs = int(gin.query_parameter('ICU_loader_semi_temporal.batch_size')) * (args.percentage/100)
        gin.clear_config()
        if not isinstance(args.seed, list):
            seeds = [args.seed]
        else:
            seeds = args.seed
        split_seeds = args.split_seed
        for split_seed in split_seeds:
            if not os.path.exists('tmp/'):
                os.mkdir('tmp/')
            fraction_data_path = build_few_label_dataset(data_dir, 'tmp/', percentage=args.percentage, seed=split_seed, task=task, overwrite=False)
            gin_bindings += ['ICU_loader_semi_temporal.data_path = '+"'"+fraction_data_path+"'"]
            scaled_bs = int(scaled_bs)
            gin_bindings += ['ICU_loader_semi_temporal.batch_size = '+ str(scaled_bs)]
            for seed in seeds:
                log_dir_seed = os.path.join(log_dir,str(seed))
                train_dir = os.path.join(log_dir_seed, "train")
                if args.eval_name :
                    eval_dir = os.path.join(log_dir_seed, args.eval_name[0] + str(args.percentage) + '_p')
                else:
                    eval_dir = os.path.join(log_dir_seed, "eval" + str(args.percentage) + '_p')
                eval_dir = os.path.join(eval_dir, str(split_seed))
                eval_gin_files = [args.train_config, args.eval_config[0]]
                gin_bindings += ['load_representation.model_dir = '+"'"+str(train_dir)+"'"]
                eval_with_gin(gin_config_files=eval_gin_files, strategy_name=args.strategy, model_dir=eval_dir,
                              overwrite=args.overwrite, seed=seed, gin_bindings=gin_bindings)
