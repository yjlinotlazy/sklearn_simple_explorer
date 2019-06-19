import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import datetime


SEED = 10
TESTSIZE = 0.33
SCORING = "accuracy"


class Timer:
    def __init__(self, message):
        self.msg = message

    def __enter__(self):
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        self.end = datetime.datetime.now()
        duration = self.end - self.start
        self.seconds = duration.seconds
        self.microseconds = duration.microseconds
        print('{}_took {} seconds, or {} microseconds'.format(self.msg, self.seconds, self.microseconds))


class Evaluator:
    """
    This is the class that does all the model evaluation and exports results as
    figures. It assumes that the data has been processed. In particular,
    the data should consists of features and labels, where the last column stores
    the label.
    """
    def __init__(self, datapath, dataname, classification="binary", save_fig="."):
        """
        Parameters
        ----------
        datapath : str
            Path of the data. The data should be in csv with no index and with header
        dataname : str
            name of the dataset. This will be used as labels on the figures
        classification : str
            Whether it's binary or multiclass classification.
        save_fig : str
            Folder of where to save figures
        """
        self.raw_df = pd.read_csv(datapath)
        self.dataname = dataname
        self.df_train, self.df_test = train_test_split(self.raw_df,
                                                       test_size=TESTSIZE,
                                                       random_state=SEED)
        self.fig_path = save_fig
        self.clf = None
        self.param_grid = None
        self.best_clf = None
        self.train_sizes, self.train_mean, self.val_mean = None, None, None

    def plot_labels(self):
        """Get a sense of distribution of labels"""
        # self.raw_df.iloc[:, -1].hist()
        plt.hist(self.raw_df.iloc[:, -1])
        plt.title("Distribution of classes")
        plt.xlabel("Class labels of {}".format(self.dataname))
        plt.ylabel("Count")
        plt.savefig('{}/{}_classes.png'.format(self.fig_path, self.dataname))
        print("Class frequencies")
        v_counts = self.raw_df.iloc[:, -1].value_counts()
        print(v_counts)
        print(max(v_counts) / v_counts.sum())

    def set_classifier(self, clf, para_dist):
        """Set the classifier with parameter space"""
        self.clf = clf
        self.param_grid = para_dist

    def grid_search(self, n_iter=10, cv=5):
        random_search = GridSearchCV(self.clf, param_grid=self.param_grid,
                                     scoring=SCORING,
                                     cv=cv,
                                     return_train_score=True)
        random_search.fit(self.df_train.iloc[:, :-1], self.df_train.iloc[:, -1])
        # assign best classifier for further evaluation
        with Timer('Grid search'):
            self.best_clf = random_search.best_estimator_

    def plot_learning_curve(self, ylim=(0.7, 1.01), cv=5,
                            train_sizes=np.linspace(.1, 1.0, 5),
                            title=None):
        # 95% copied from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
        # with some modifications
        plt.figure()
        if not title:
            title = self.best_clf.__class__.__name__
        plt.title('Learning curve for {}'.format(title))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples for {}".format(self.dataname))
        plt.ylabel(SCORING)

        train_sizes, train_scores, test_scores = learning_curve(
            self.best_clf, self.df_train.iloc[:, :-1],
            self.df_train.iloc[:, -1], cv=cv, scoring=SCORING, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        save_path = os.path.join(self.fig_path, self.dataname + self.best_clf.__class__.__name__ +
                                 "_LC.png")
        if title:
            save_path = os.path.join(self.fig_path, self.dataname + title + '_LC.png')
        plt.savefig(save_path)
        self.train_sizes, self.train_mean, self.val_mean = train_sizes,\
                                                           train_scores_mean,\
                                                           test_scores_mean

    def report(self):
        # Find the trianing size where train and val scores has min diff
        print("Params of best model")
        print(self.best_clf)
        diff = self.train_mean - self.val_mean
        # of all the argmins, take the larger training size
        argmin = [i for i in range(len(diff)) if diff[i] == min(diff)][-1]
        print("train set size for min diff between cross val and train:", self.train_sizes[argmin])

        # Find the training size where val score is the highest
        argmax = [i for i in range(len(self.val_mean)) if self.val_mean[i] == max(self.val_mean)][-1]
        print("train set size for highest cross val score:", self.train_sizes[argmax])
        with Timer("Training the best model"):
            self.best_clf.fit(self.df_train.iloc[:, :-1], self.df_train.iloc[:, -1])
        test_truth = self.df_test.iloc[:, -1].values
        test_pred = self.best_clf.predict(self.df_test.iloc[:, :-1])
        print("Test accuracy", accuracy_score(y_true=test_truth, y_pred=test_pred))
        print(classification_report(y_true=test_truth, y_pred=test_pred))