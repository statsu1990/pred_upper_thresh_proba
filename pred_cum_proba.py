import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as sklm
from sklearn import ensemble as skes

def make_x_y(n1, n2, thresh_range, df=3):
    """
    Returns:
        x : [t, thresh]
        y : thresh > t
    """
    # random sampling from standard t dist
    t = np.random.standard_t(df, [n1,1])
    z = t

    x, y = [], []
    for i in range(n2):
        thresh = np.random.uniform(thresh_range[0], thresh_range[1], [n1,1])
        x.append(np.concatenate([t,thresh], axis=1))
        y.append((thresh > z).astype('int'))
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y

def eval_v1(clf, tresh_range, n_t=6, n_thresh=50, title='', filename=''):
    z = np.linspace(tresh_range[0], tresh_range[1], n_t)
    thresh = np.linspace(tresh_range[0], tresh_range[1], n_thresh)[:,None]

    for i in range(n_t):
        x = np.concatenate([[[z[i]]]*n_thresh, thresh], axis=1)
        proba = clf.predict_proba(x)[:,1]
        plt.plot(thresh, proba, label=f'pred proba z>{z[i]}')
        plt.axvline(z[i], label=f'thresh={z[i]}', color=f'{i/n_t}')
    plt.xlabel('thresh')
    plt.ylabel('pred probability of z > thresh')
    plt.title(title)
    plt.legend()
    plt.savefig(filename+'upper_thresh_proba.png')

def main_v1():
    RANDOM_STATE = 2020
    np.random.seed(RANDOM_STATE)
    N1, N2 = 1000, 5
    DF = 3
    TRESH_RANGE = (-5,5)

    tr_x, tr_y = make_x_y(N1, N2, TRESH_RANGE, DF)
    ts_x, ts_y = make_x_y(N1, N2, TRESH_RANGE, DF)

    # clf : predict cumulative probability of t
    clf = sklm.LogisticRegression(random_state=RANDOM_STATE)
    clf.fit(tr_x, tr_y)
    print(f'tr score {clf.score(tr_x, tr_y)}, ts score {clf.score(ts_x, ts_y)}')

    eval_v1(clf, TRESH_RANGE, title='z = sampled by t dist', filename='t_dist_')
    return

def make_x_y_v2(n1, n2, thresh_range, df=3):
    """
    Returns:
        x : [t, thresh]
        y : thresh > t
    """
    # random sampling from standard t dist
    temp_x = np.random.normal(size=(n1,2))
    z = 0.5 * temp_x[:,0] + 1.0 * temp_x[:,1] + np.random.normal(size=(n1,)) * 0.3
    z = z[:,None]

    x, y = [], []
    for i in range(n2):
        thresh = np.random.uniform(thresh_range[0], thresh_range[1], [n1,1])
        x.append(np.concatenate([temp_x, thresh], axis=1))
        y.append((thresh > z).astype('int'))
    x = np.concatenate(x)
    y = np.concatenate(y)
    return x, y

def eval_v2(clf, tresh_range, n_t=6, n_thresh=50, title='', filename=''):
    temp_x = np.array([[-2,-2], [-2,0], [-2,2], [-1.5,0.5], [0.5,0.5], [1.5,0.5], ])
    z = 0.5 * temp_x[:,0] + 1.0 * temp_x[:,1]
    z = z[:,None]
    thresh = np.linspace(tresh_range[0], tresh_range[1], n_thresh)[:,None]

    for i in range(n_t):
        a = np.repeat((temp_x[i])[None,:], len(thresh), axis=0)
        x = np.concatenate([a, thresh], axis=1)
        proba = clf.predict_proba(x)[:,1]
        plt.plot(thresh, proba, label=f'pred proba z>{z[i]}')
        plt.axvline(z[i], label=f'thresh={z[i]}', color=f'{i/n_t}')
    plt.xlabel('thresh')
    plt.ylabel('pred probability of z > thresh')
    plt.title(title)
    plt.legend()
    plt.savefig(filename+'upper_thresh_proba.png')

def main_v2():
    RANDOM_STATE = 2020
    np.random.seed(RANDOM_STATE)
    N1, N2 = 1000, 5
    TRESH_RANGE = (-5,5)

    tr_x, tr_y = make_x_y_v2(N1, N2, TRESH_RANGE)
    ts_x, ts_y = make_x_y_v2(N1, N2, TRESH_RANGE)

    # clf : predict cumulative probability of t
    clf = sklm.LogisticRegression(random_state=RANDOM_STATE)
    clf.fit(tr_x, tr_y)
    print(f'tr score {clf.score(tr_x, tr_y)}, ts score {clf.score(ts_x, ts_y)}')

    eval_v2(clf, TRESH_RANGE, title='z = 0.5*N(0,1)+1*N(0,1)+ε', filename='linear_')
    return



if __name__ == '__main__':
    #main_v1()
    main_v2()