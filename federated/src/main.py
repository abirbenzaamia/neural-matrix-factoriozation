from fedanal.plot import Plot
from fedanal.predict import predict_all_clients, generate_frequency_in_topk
from fedanal.triehh import SimulateTrieHH
from server import run_server, initialize_clients
from federeco.train import sample_clients
from dataset import Dataset
import argparse
from pathlib import Path

from federeco.config import MODEL_PARAMETERS, LEARNING_RATE, BATCH_SIZE, TOPK


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='federeco',
        description='federated recommendation system',
    )
    params = MODEL_PARAMETERS['FedNCF']
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument('-lr', '--learning_rate', default=LEARNING_RATE, metavar='learning-rate',
                        help='learning rate value for model training')
    parser.add_argument('-lf', '--latent_factors', default=params['mf_dim'], metavar='learning-rate',
                        help='number of latent factors of user and item matrices')
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, metavar='batch-size',
                        help='batch size for local model at the user-level')
    parser.add_argument('-k', '--top_k', default=TOPK, metavar='batch-size',
                        help='batch size for local model at the user-level')
    parser.add_argument('-n', '--name', default='FedNeuMF', metavar='name',
                        help='name of the model')
    parser.add_argument('-d', '--dataset', default='movielens', metavar='dataset',
                        choices=['movielens', 'amazon', 'foursquare'],
                        help='which dataset to use, default "movielens"')
    parser.add_argument('-p', '--path', default='../../dataset', metavar='path',
                        help='path where trained model is stored, default "pretrained/ncf.h5"')
    parser.add_argument('-e', '--epochs', default=400, metavar='epochs', type=int,
                        help='number of training epochs, default 500')
    parser.add_argument('-s', '--save', default=True, action='store_true',
                        help='flag that indicates if trained model should be saved')
    parser.add_argument('-c', default=20, metavar='sample_size', type=int,
                        help='number of clients to sample per epoch')
    parser.add_argument('-l', default=3, metavar='local_epochs', type=int,
                        help='number of local training epochs')
    parser.add_argument('-v', '--validation_steps', default=10, type=int)
    args, leftovers = parser.parse_known_args()
    if args.dataset == 'movielens': 
        parser.add_argument(
        "--type",
        type=str,
        default="ml-100k",
        choices=["ml-1m", "ml-25m", "ml-100k"],
        help="decide which type of movielens dataset: ml-1m, ml-25m or ml-100k",)
    if args.dataset == 'amazon': 
        parser.add_argument(
        "--type",
        type=str,
        default="grocery",
        choices=["grocery", "ml-25m", "ml-100k"],
        help="decide which type of Amazon products",)
    if args.dataset == 'foursquare': 
        parser.add_argument(
        "--type",
        type=str,
        default="nyc",
        choices=["nyc", "tky"],
        help="decide which type of foursquare dataset",)
    return parser.parse_args()


def main():
    args = parse_arguments()
    raw_path = Path(args.path) / args.dataset / args.type
    #raw_path = Path(args.path) / args.dataset 
    out_path = 'pretrained/%s/%s/%s_FedNeuMF_c%d.h5' %(args.dataset, args.type, args.type, args.c)
    # instantiate the dataset based on passed argument
    dataset = Dataset(raw_path)
    # run the server to load the existing model or train & save a new one
    trained_model = run_server(dataset, num_clients=args.c, epochs=args.epochs,
                               path=out_path, save=args.save, local_epochs=args.l, args=args)
    # pick random client & generate recommendations for them
    clients = initialize_clients(dataset)
    client, _ = sample_clients(clients, dataset.num_users)
#     recommendations = client[1].generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k = args.top_k)
#     hist = client[1].get_historical_data()

#     print('Recommendations for user id:', client[1].client_id)
#     if args.dataset == 'movielens':
#         print(dataset.get_movie_names(recommendations))
    
#     else:
#         print(recommendations)

#     print('Historical data for user id:', client[1].client_id)
#     print(dataset.get_movie_names(hist))

# if __name__ == '__main__':
#     main()
    out_path = 'pretrained'
    predictions = predict_all_clients(trained_model, client,dataset.num_users, dataset.num_items, out_path)

    max_k = 10

    # length of longest word
    max_word_len = 10
    # epsilon for differential privacy
    epsilon = 4
    # delta for differential privacy
    delta = 2e-8
    
    # repeat simulation for num_runs times
    num_runs = 10
    final_result = []
    final_freq = []
    print('--------------------- Federated analytics ---------------------')
    for topk in range(max_k):
        simulate_triehh = SimulateTrieHH(
        out_path= out_path, max_list_len=max_word_len, epsilon=epsilon, delta=delta, num_runs=num_runs)
        triehh_heavy_hitters = simulate_triehh.get_heavy_hitters(topk+1)
        print('Discovered top items in top,', topk+1)
        final_result.append(triehh_heavy_hitters)
 

    print('--------------------- Evaluation ---------------------')
    plot = Plot(max_k)
    precisions = []
    recalls = []
    f1_scores = []
    for topk in range(max_k):
        # get real frequencies 
        freq = generate_frequency_in_topk(predictions, topk+1, out_path)
        #print('-----------',freq)
        discovered_pop_items = final_result[topk]
        precision, recall, f1_score = plot.calculate_f1_score(discovered_pop_items, freq)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    
    plot.plot_f1_scores(precisions, recalls, f1_scores, epsilon, max_k=max_k)
    # recommendations = client[1].generate_recommendation(server_model=trained_model, num_items=dataset.num_items, k = args.top_k)
    # hist = client[1].get_historical_data()
    
    # print('Recommendations for user id:', client[1].client_id)
    # if args.dataset == 'movielens':
    #     print(dataset.get_movie_names(raw_path,recommendations))
    
    # else:
    #     print(recommendations)

    # print('Historical data for user id:', client[1].client_id)
    # print(dataset.get_movie_names(raw_path, hist))
    #federated analytics
    # print('--------------------- Federated analytics ---------------------')
    # get_similar_items(trained_model, dataset.num_users, dataset.num_items, k=10, dataset=dataset, raw_path = raw_path)
if __name__ == '__main__':
    main()
