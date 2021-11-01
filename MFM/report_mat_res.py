'''
Module containing the functions that create tables and plots for scientific articles.
'''


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


def plot_k_dist_1D(rows, cov, k, title, y, axes):
    '''
    Plots distributions of k x 1 vectors, depending on k.
    Rows is an n x k array.
    cov is a flattened kxk covariance matrix of the rows
    '''
    cov_mat = []
    for i in range(k):
        cov_mat.append(cov[i*k: (i+1)*k])

    cov_mat = np.matrix(cov_mat)  # get covariance matrix
    print("Estimated Covariance Matrix")
    print(cov_mat)
    sig_half_inv = np.matrix(LA.inv(LA.sqrtm(cov_mat)))
    print(rows.shape)
    rows_t = sig_half_inv * rows.transpose()  # standardize!
    rows = np.array(rows_t.transpose())
    print("Standardized Variances")
    print(np.var(rows, axis=0))

    if k == 1:
        plt.sca(axes[y*3])
        sns.distplot(rows, kde=True)
        axes[y*3].set_title(title, fontsize=12)

        axes[y*3+1].set_visible(False)
        axes[y*3+2].set_visible(False)

    elif k == 2:

        plt.sca(axes[y*3+1])
        sns.distplot(rows[:, 0], kde=True)
        axes[y*3].set_title("1", fontsize=12)

        plt.sca(axes[y*3+1])
        sns.distplot(rows[:, 1], kde=True)
        axes[y*3+1].set_title("2", fontsize=12)

        axes[y*3+2].set_visible(False)

    else:  # k == 3:
        x_min = min(rows[:, 0])
        x_max = max(rows[:, 0])

        x_norm = np.linspace(x_min, x_max, 100)
        y_norm = scipy.stats.norm.pdf(x_norm, 0, 1)

        plt.sca(axes[y*3])
        sns.distplot(rows[:, 0], kde=False, norm_hist=True)
        axes[y*3].plot(x_norm, y_norm, color='coral')
        axes[y*3].set_title("1", fontsize=12)
        axes[y*3].set_ylim([0, max(y_norm) + 0.10])
        axes[y*3].set_xlim([-5., 5.])
        x_min = min(rows[:, 1])
        x_max = max(rows[:, 1])

        x_norm = np.linspace(x_min, x_max, 100)
        y_norm = scipy.stats.norm.pdf(x_norm, 0, 1)

        plt.sca(axes[y*3+1])
        sns.distplot(rows[:, 1], kde=False, norm_hist=True)
        axes[y*3+1].plot(x_norm, y_norm, color='coral')
        axes[y*3+1].set_ylim([0, max(y_norm) + 0.10])
        axes[y*3+1].set_xlim([-5., 5.])
        axes[y*3+1].set_title("2", fontsize=12)

        x_min = min(rows[:, 2])
        x_max = max(rows[:, 2])

        x_norm = np.linspace(x_min, x_max, 100)
        y_norm = scipy.stats.norm.pdf(x_norm, 0, 1)

        plt.sca(axes[y*3+2])
        sns.distplot(rows[:, 2], kde=False, norm_hist=True)
        axes[y*3+2].plot(x_norm, y_norm, color='coral')
        axes[y*3+2].set_ylim([0, max(y_norm) + 0.10])
        axes[y*3+2].set_xlim([-5., 5.])
        axes[y*3+2].set_title("3", fontsize=12)


def superplot_k_dist(errR, covR, supertitle):
    k1, k2 = dim_latent
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 4 horizontal
    fig.suptitle(supertitle)
    titles = ["p,q = 50,50"]
    for i in range(len(titles)):
        plot_k_dist(errR[i], covR[i], k1, "" + titles[i], i, axes)
    fig.savefig("testing.png")


def QQ_plot_1D(rows, k, title, y, axes):
    '''
    Plots QQplots of k x 1 vectors, depending on k.
    Rows is an n x k array, to show normality
    '''
    if k == 1:
        stats.probplot(rows, dist="norm", plot=pylab)
        axes[y*3].set_title(title, fontsize=12)
        axes[y*3+1].set_visible(False)
        axes[y*3+2].set_visible(False)
    elif k == 2:

        plt.sca(axes[y*3+1])
        stats.probplot(rows[:, 0], dist="norm", plot=pylab)
        axes[y*3].set_title("1", fontsize=12)

        plt.sca(axes[y*3+1])
        stats.probplot(rows[:, 1], dist="norm", plot=pylab)
        axes[y*3+1].set_title("2", fontsize=12)

        axes[y*3+2].set_visible(False)

    else:  # k == 3:

        plt.sca(axes[y*3])
        stats.probplot(rows[:, 0], dist="norm", plot=pylab)
        axes[y*3].set_title("1", fontsize=12)

        plt.sca(axes[y*3+1])
        stats.probplot(rows[:, 1], dist="norm", plot=pylab)
        axes[y*3+1].set_title("2", fontsize=12)

        plt.sca(axes[y*3+2])
        stats.probplot(rows[:, 1], dist="norm", plot=pylab)
        axes[y*3+2].set_title("3", fontsize=12)


def superplot_k_1D(errR, covR, supertitle):
    k1, k2 = dim_latent
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 4 horizontal
    fig.suptitle(supertitle)
    # ["p,q = 20,20","p,q= 20,40","p,q = 40,20","p,q = 40,40"]
    titles = ["p,q = 50,50"]
    for i in range(len(titles)):
        plot_k_dist_1D(errR[i], covR[i], k1, "" + titles[i], i, axes)
    fig.savefig("testing1D.png")


def superplot_QQ_1D(errR, supertitle):
    k1, k2 = dim_latent
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 4 horizontal
    fig.suptitle(supertitle)
    titles = ["p,q = 50,50"]
    for i in range(len(titles)):
        QQ_plot_1D(errR[i],  k1, "" + titles[i], i, axes)
    fig.savefig("testingQQ1D.png")


def reporter_chen(setting):
    '''
    (M4) Reporter
    Creates boxplots  to evaluate the matrix factor model & how 
    closely the loading spaces are estimated when we vary given parameters.
    Plots are saved to png.
    '''
    '''
  For all plots: k1, k2 = 3, 2
                 n = 200
                 h0 = 1
                 Ecov = (None,None) # identity
                 lam = 1
  Vary:
  deltaR, deltaC = (0,0), (0,0.5), (0.5,0.5)
  T = 0.5pq, pq, 1.5pq, 2pq
  (p, q) = (20,20), (100,20), (20,100), (100,100)
  '''
    print("deltas = 0,0")
    frames, framesF, err, covR = get_dfs_chen(
        setting)  # change later to frames, framesF, err
    #save_plot_dfs(frames, "Loading Space Estimate Distances", "00plot.png")
    #save_plot_dfs_F(framesF, "$||\hat{F}_t - H_R^{-1} F_t H_C^{-1T}||$", "00Fplot.png")
    supertitles = ["T = 100", "T = 50", "T = 25"]
    for i in range(len(err)):
        superplot_k_dist(err[i], covR[i], supertitles[i])
        superplot_k_1D(err[i], covR[i], supertitles[i])
        superplot_QQ_1D(err[i], supertitles[i])

    return (frames, framesF, err, covR)
