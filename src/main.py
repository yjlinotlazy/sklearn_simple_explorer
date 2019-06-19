from evaluate import Evaluator
from model_params import PARAMETERS
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import argparse
import warnings
warnings.filterwarnings("ignore")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("datapath", help="path of the data saved as csv")
    argparser.add_argument("savepath", help="path to save figures")
    argparser.add_argument('dataname', help="name of the dataset to get parameters. Must be ctg or wine")
    args = argparser.parse_args()
    evaluator = Evaluator(args.datapath, args.dataname, save_fig=args.savepath)

    if args.dataname not in ['ctg', 'wine']:
        raise ValueError("Please enter either ctg or wine as dataname.")
    if args.dataname == 'ctg':
        ylim = (0.7, 1.01)
    else:
        ylim = (0.3, 1.01)
    params = PARAMETERS[args.dataname]

    evaluator.plot_labels()

    # Evaluate decision tree
    clf = DecisionTreeClassifier()
    param_grid = params['tree']
    print("Parameters for random search")
    print(param_grid)
    evaluator.set_classifier(clf, param_grid)
    evaluator.grid_search()
    evaluator.plot_learning_curve(ylim=ylim)
    evaluator.report()

    # Pre-pruning by limiting leaf size, all other parameters identical
    clf = DecisionTreeClassifier(min_weight_fraction_leaf=0.05, max_leaf_nodes=20)
    param_grid = params['tree']
    print("With pruning")
    evaluator.set_classifier(clf, param_grid)
    evaluator.grid_search()
    evaluator.plot_learning_curve(title='pruned', ylim=ylim)
    evaluator.report()

    # Boosting
    base_clf = DecisionTreeClassifier(min_weight_fraction_leaf=0.05, max_leaf_nodes=20)
    # base_clf = DecisionTreeClassifier()
    clf = AdaBoostClassifier(base_clf)
    param_grid = params['boosting']
    print("Parameters for Boosting")
    print(param_grid)
    evaluator.set_classifier(clf, param_grid)
    evaluator.grid_search()
    evaluator.plot_learning_curve(ylim=ylim)
    evaluator.report()

    # KNN
    clf = KNeighborsClassifier()
    param_grid = params['knn']
    print("Parameters for KNN")
    print(param_grid)
    evaluator.set_classifier(clf, param_grid)
    evaluator.grid_search()
    evaluator.plot_learning_curve(ylim=ylim)
    evaluator.report()

    # SVM
    clf = SVC()
    param_grid = params['svm']
    print("Parameters for SVM")
    print(param_grid)
    evaluator.set_classifier(clf, param_grid)
    evaluator.grid_search()
    evaluator.plot_learning_curve(ylim=ylim)
    evaluator.report()

    # NN
    clf = MLPClassifier(max_iter=10000000000)
    param_grid = params['nn']
    print("Parameters for NN")
    print(param_grid)
    evaluator.set_classifier(clf, param_grid)
    evaluator.grid_search()
    evaluator.plot_learning_curve(ylim=ylim)
    evaluator.report()


if __name__ == '__main__':
    main()

