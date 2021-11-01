import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import linalg as LA
import scipy

def plot_k_dist_1D(rows, k, title, y, axes):
  '''
  Plots distributions of k x 1 vectors, depending on k.
  Rows is an n x k array.
  cov is a flattened kxk covariance matrix of the rows
  '''

  if k == 1:
    plt.sca(axes[y*3])
    sns.distplot(rows, kde=True)
    axes[y*3].set_title(title, fontsize=12)
    
    axes[y*3+1].set_visible(False)
    axes[y*3+2].set_visible(False)
    
  elif k == 2:
    
    plt.sca(axes[y*3+1])
    sns.distplot(rows[:,0], kde=True)
    axes[y*3].set_title("1", fontsize=12)
    
    plt.sca(axes[y*3+1])
    sns.distplot(rows[:,1], kde=True)
    axes[y*3+1].set_title("2", fontsize=12)
    
    axes[y*3+2].set_visible(False)
    
  else: # k == 3:
    x_min = min(rows[:,0])
    x_max = max(rows[:,0])
    
    x_norm = np.linspace(x_min, x_max, 100)
    y_norm = scipy.stats.norm.pdf(x_norm,0,1)
    
    plt.sca(axes[y*3])
    sns.distplot(rows[:,0], kde=False, norm_hist=True, bins=100)
    axes[y*3].plot(x_norm, y_norm, color='coral')
    axes[y*3].set_title("1", fontsize=12)
    axes[y*3].set_ylim([0, max(y_norm) + 0.10])
    axes[y*3].set_xlim([-5., 5.])
    x_min = min(rows[:,1])
    x_max = max(rows[:,1])
    
    x_norm = np.linspace(x_min, x_max, 100)
    y_norm = scipy.stats.norm.pdf(x_norm,0,1)
    
    plt.sca(axes[y*3+1])
    sns.distplot(rows[:,1], kde=False, norm_hist=True, bins=100)
    axes[y*3+1].plot(x_norm, y_norm, color='coral')
    axes[y*3+1].set_ylim([0, max(y_norm) + 0.10])
    axes[y*3+1].set_xlim([-5., 5.])
    axes[y*3+1].set_title("2", fontsize=12)
    
    x_min = min(rows[:,2])
    x_max = max(rows[:,2])
    
    x_norm = np.linspace(x_min, x_max, 100)
    y_norm = scipy.stats.norm.pdf(x_norm,0,1)
    
    plt.sca(axes[y*3+2])
    sns.distplot(rows[:,2], kde=False, norm_hist=True, bins=100)
    axes[y*3+2].plot(x_norm, y_norm, color='coral')
    axes[y*3+2].set_ylim([0, max(y_norm) + 0.10])
    axes[y*3+2].set_xlim([-5., 5.])
    axes[y*3+2].set_title("3", fontsize=12)


def plot_k_dist(rows, cov, k, title, y, axes):
    '''
    Plots distributions of k x 1 vectors, depending on k.
    Rows is an n x k array.
    cov is a flattened kxk covariance matrix of the rows
    '''
    cov_mat = []
    for i in range(k):
        cov_mat.append(cov[i*k: (i+1)*k])
    cov_mat = np.matrix(cov_mat)  # get covariance matrix
    sig_half_inv = np.matrix(LA.inv(LA.sqrtm(cov_mat)))
    rows_t = sig_half_inv * rows.transpose()  # standardize!
    rows = np.array(rows_t.transpose())
    if k == 1:
        plt.sca(axes[y*3])
        sns.distplot(rows, kde=True)
        axes[y*3].set_title(title, fontsize=12)
        axes[y*3+1].set_visible(False)
        axes[y*3+2].set_visible(False)
    elif k == 2:

        axes[y*3].scatter(rows[:, 0], rows[:, 1])

        axes[y*3].set_title(title)
        axes[y*3].set_xlabel('1')
        axes[y*3].set_ylabel('2')

        axes[y*3+1].set_visible(False)
        axes[y*3+2].set_visible(False)

    else:  # k == 3:

        axes[y*3].scatter(rows[:, 0], rows[:, 1],
                          facecolors='none', edgecolors='black')

        axes[y*3].set_title(title)
        axes[y*3].set_xlabel('1')
        axes[y*3].set_ylabel('2')

        axes[y*3+1].scatter(rows[:, 1], rows[:, 2],
                            facecolors='none', edgecolors='black')

        axes[y*3+1].set_xlabel('2')
        axes[y*3+1].set_ylabel('3')

        axes[y*3+2].scatter(rows[:, 0], rows[:, 2],
                            facecolors='none', edgecolors='black')

        axes[y*3+2].set_xlabel('1')
        axes[y*3+2].set_ylabel('3')
