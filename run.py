import argparse
from data import *
import pickle

from ICL_LLM import CorrectionShift_ICL,SimulatedData_choose_lambda,SimulatedData_ICL,TemporalShift_ICL,GeospatialShift_ICL,CorrectionShift_choose_lambda,TemporalShift_choose_lambda,GeospatialShift_choose_lambda
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=5, help='number of trials to run/experiment')
    parser.add_argument('--data', default="correction", help='which dataset to use')
    parser.add_argument('--api_key', default="", help='which API_KEY to use')
    parser.add_argument('--base_url', default="", help='which url to use')
    parser.add_argument('--base_model', default='', help='which model to use')
    parser.add_argument('--numshots', default=100, help='the number of shots to use')
    parser.add_argument('--test_num', default=100, help='the number of demos to test')
    parser.add_argument('--cost', default="l1", help='which cost fn to use')
    parser.add_argument('--recourse', default="ZOO", help='which recourse approach to use')
    args = parser.parse_args()

    result_fname = "_".join([args.data, args.base_model, args.cost, args.recourse, str(args.n_trials)]) + ".pkl"

    results = {}

    for i in range(args.n_trials):
        print("Trial %d" % i)

        results_i = {}
        fold = i

        print("Loading %s dataset" % args.data)
        if args.data == "correction":
            data = CorrectionShift(fold)
            data1, data2 = data.get_data("dataset/german.csv", "dataset/corrected_german.csv")
            mean1, var1, mean2, var2 = data.get_mean_var()

        elif args.data == "temporal":
            data = TemporalShift(fold)
            data1, data2 = data.get_data("dataset/SBAcase.11.13.17.csv")
            mean1, var1, mean2, var2 = data.get_mean_var()

        elif args.data == "geospatial":
            data = GeospatialShift(fold)
            data1, data2 = data.get_data("dataset/student-por.csv", sep=";")
            mean1, var1, mean2, var2 = data.get_mean_var()

        elif args.data == "SimulatedData":
            data = SimulatedData(fold)
            data1 = data.get_data()
        X1_train, y1_train, X1_test, y1_test = data1
        # X2_train, y2_train, X2_test, y2_test = data2

        print("LLM: %s " % args.base_model)
        print(f"numshots:{args.numshots}")
        if args.data == "correction":
            X1_acc, X1_train_testnum_prediction, X1_prompt = CorrectionShift_ICL(args.numshots,args.test_num, X1_train, y1_train, X1_test, y1_test,args.base_model, args.api_key, args.base_url)
            # X2_acc, X2_train_testnum_prediction, X2_prompt = CorrectionShift_ICL(args.numshots,args.test_num, X2_train, y2_train, X2_test, y2_test,args.base_model, args.api_key, args.base_url)
        if args.data == "temporal":
            X1_acc, X1_train_testnum_prediction, X1_prompt = TemporalShift_ICL(args.numshots,args.test_num, X1_train, y1_train, X1_test, y1_test,args.base_model, args.api_key, args.base_url)
            # X2_acc, X2_train_testnum_prediction = TemporalShift_ICL(args.numshots,args.test_num, X2_train, y2_train, X2_test, y2_test,args.base_model, args.api_key, args.base_url)
        if args.data == "geospatial":
            X1_acc, X1_train_testnum_prediction, X1_prompt = GeospatialShift_ICL(args.numshots,args.test_num, X1_train, y1_train, X1_test, y1_test,args.base_model, args.api_key, args.base_url)
            # X2_acc, X2_train_testnum_prediction = GeospatialShift_ICL(args.numshots,args.test_num, X2_train, y2_train, X2_test, y2_test,args.base_model, args.api_key, args.base_url)
        if args.data == "SimulatedData":
            X1_acc, X1_train_testnum_prediction, X1_prompt = SimulatedData_ICL(args.numshots,args.test_num, X1_train, y1_train, X1_test, y1_test,args.base_model, args.api_key, args.base_url)
            # X2_acc, X2_train_testnum_prediction = SimulatedData_ICL(args.numshots,args.test_num, X2_train, y2_train, X2_test, y2_test,args.base_model, args.api_key, args.base_url)

        results_i["X1_acc"] = X1_acc
        # results_i["X2_acc"] = X2_acc

        print(f'X1_acc: {X1_acc}')
        # print(f'X2_acc: {X2_acc}')

        print("Finding where recourse is needing on X1_test")
        target = 1
        recourse_needed_idx_X1_testnum_train = np.where(X1_train_testnum_prediction == 1 - target)[0]
        recourse_needed_X1_train = X1_train.iloc[recourse_needed_idx_X1_testnum_train].values

        print("Using %s cost" % args.cost)

        print("Getting %s recourse" % args.recourse)
        if args.data == "correction":
            lamb, recourse, validity, L1_loss = CorrectionShift_choose_lambda(X_old=recourse_needed_X1_train, target=target,
                                                                     mean=mean1, var=var1, prompt=X1_prompt,
                                                                     MODEL_NAME=args.base_model, API_KEY=args.api_key,
                                                                     BASE_URL=args.base_url)

        elif args.data == "temporal":
            lamb, recourse, validity, L1_loss = TemporalShift_choose_lambda(X_old=recourse_needed_X1_train, target=target,
                                                                     mean=mean1, var=var1, prompt=X1_prompt,
                                                                     MODEL_NAME=args.base_model, API_KEY=args.api_key,
                                                                     BASE_URL=args.base_url)


        elif args.data == "geospatial":
            lamb, recourse, validity, L1_loss = GeospatialShift_choose_lambda(X_old=recourse_needed_X1_train, target=target,
                                                                     mean=mean1, var=var1, prompt=X1_prompt,
                                                                     MODEL_NAME=args.base_model, API_KEY=args.api_key,
                                                                     BASE_URL=args.base_url)
        elif args.data == "SimulatedData":
            lamb, recourse, validity, L1_loss = SimulatedData_choose_lambda(X_old=recourse_needed_X1_train, target=target,
                                                                     prompt=X1_prompt,MODEL_NAME=args.base_model, API_KEY=args.api_key,
                                                                     BASE_URL=args.base_url)

        results_i["recourses"] = recourse
        results_i["X1_validity"] = validity
        results_i["lambda"] = lamb
        results_i["cost"] = L1_loss
        results[i] = results_i
        print("X1 validity: %f" % validity)
        print("lambda: %f" % lamb)
        print("cost: %f" % L1_loss)

    with open(result_fname, "wb") as f:
        pickle.dump(results, f)

    agg_X1_validity = []
    agg_cost = []
    agg_acc = []
    for i in range(args.n_trials):
        if results[i]['X1_validity'] == None:
            continue
        agg_X1_validity.append(results[i]["X1_validity"])
        # agg_m2_validity.append(results[i]["m2_validity"])
        agg_cost.append(results[i]["cost"])
        agg_acc.append(results[i]["X1_acc"])
    print("Average X1 acc: %f +- %f" % (np.mean(agg_acc), np.std(agg_acc)))
    print("Average X1 validity: %f +- %f" % (np.mean(agg_X1_validity), np.std(agg_X1_validity)))
    # print("Average M2 validity: %f +- %f" % (np.mean(agg_m2_validity), np.std(agg_m2_validity)))
    print("Average cost: %f +- %f" % (np.mean(agg_cost), np.std(agg_cost)))








