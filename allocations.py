import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm
import scipy.stats

class ReturnDistribution(object):
    def price_to_return(self, price):
        return 100.0 * (price/self.current_price-1.0)
    
    def return_to_price(self, return_value):
        return self.current_price * (return_value/100.0 + 1.0)

    def initialize_from_projected_price_distribution(self, price_distribution):
        self.labels = np.zeros(price_distribution.num_bins, dtype=np.float)
        self.pdf_function = lambda x: price_distribution.pdf_function(self.return_to_price(x))
        self.pdf_sampler = lambda n: self.price_to_return(price_distribution.pdf_sampler(n))
        self.num_bins = price_distribution.num_bins
        for i in range(0,self.labels.shape[0]):
            self.labels[i] = self.price_to_return(price_distribution.labels[i])
        self.likelihoods = self.pdf_function(self.labels)

    def __init__(self, current_price=None, projected_price_distribution=None, labels=None, pdf_function=None, pdf_sampler=None):
        assert (current_price is not None and projected_price_distribution is not None) ^ (labels is not None and pdf_function is not None)
        self.current_price = current_price
        if projected_price_distribution is not None:
            self.initialize_from_projected_price_distribution(projected_price_distribution)
        else:
            self.labels = labels
            self.pdf_function = pdf_function
            self.pdf_sampler = pdf_sampler
            self.likelihoods = self.pdf_function(self.labels)
    
    def __str__(self):
        ans = ""
        for n in range(0,self.num_bins):
            ans += "{}: {}\n".format(self.labels[n], self.likelihoods[n])
        return ans

    def plot(self, title="Return Distribution"):
        fig, ax = plt.subplots(1,1,tight_layout=True)
        ax.plot(self.labels, self.likelihoods)   
        plt.title(title)
        plt.show()

class PriceDistribution(object):
    def __init__(self, labels, pdf_function, pdf_sampler):
        self.pdf_function = pdf_function
        self.pdf_sampler = pdf_sampler
        self.likelihoods = self.pdf_function(labels)
        self.labels = labels
        self.num_bins = labels.shape[0] 
        
    def __str__(self):
        ans = ""
        for n in range(0,self.num_bins):
            ans += "{}: {}\n".format(self.labels[n], self.likelihoods[n])
        return ans

    def plot(self, title="Price Distribution"):
        fig, ax = plt.subplots(1,1,tight_layout=True)
        ax.plot(self.labels, self.likelihoods)   
        plt.title(title)
        plt.show()

class Token(object):
    def __init__(self, name, current_price, weight, price_distribution=None, return_distribution=None):
        assert (price_distribution==None) ^ (return_distribution==None)
        self.name = name
        self.weight = weight
        if price_distribution is not None:
            self.price_distribution = price_distribution
            self.return_distribution = ReturnDistribution(current_price, price_distribution)
        else:
            self.price_distribution = None
            self.return_distribution = return_distribution

class TokenSet(object):
    def __init__(self, token_list):
        self.token_list = token_list
        self.name_index_map = {}
        for i, token in enumerate(token_list):
            self.name_index_map[token.name] = token

    def token(self, name):
        return self.name_index_map[name]

    def plot_joint_return_distribution(self, num_samples=2**15, bins=128, xmax=1000.0):
        fix, (ax1, ax2) = plt.subplots(1, 2, sharex=True, tight_layout=True)
        return_samples = np.zeros(num_samples, dtype=np.float)
        for i, token in enumerate(self.token_list):
            return_samples += token.weight * token.return_distribution.pdf_sampler(num_samples)
        ax1.hist(return_samples, bins=bins, normed=True)
        alpha, loc, beta = scipy.stats.gamma.fit(return_samples)
        x_spc = np.linspace(-100.0, xmax, 256)
        ax2.plot(x_spc, gamma.cdf(x_spc, a=alpha, loc=loc, scale=beta))
        ax2.set_ylim(0.0, 1.0)
        plt.grid()
        plt.xlim(-100.0, xmax)
        plt.show()

    def plot_return_distributions_overlayed(self, xmin, xmax, ymin, ymax):
        fix, ax = plt.subplots(1, 1, tight_layout=True)
        plt.yscale('log')
        for token in self.token_list:
            ax.plot(token.return_distribution.labels, token.return_distribution.likelihoods, label=token.name)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.show()
    
    def plot_return_distributions_tiled(self, data_dict, num_rows, num_cols, xmin, xmax):
        fix, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=False, tight_layout=True, squeeze=True)
        for key in data_dict:
            token = self.token(key)
            index = data_dict[key]
            ax[index[0]][index[1]].plot(token.return_distribution.labels, token.return_distribution.likelihoods, label=token.name)
            ax[index[0]][index[1]].legend()
            plt.xlim(xmin, xmax)
        plt.show()

#Use case example
if __name__=="__main__":

    n_samples = 2056

    btc_price      = np.linspace(0.0, 450000.0, n_samples)
    btc_price_pdf  = lambda x: np.array(gamma.pdf(x, a=1.5, loc=0.0, scale=50000.0))
    btc_price_samp = lambda n: np.array(gamma.rvs(a=1.5, loc=0.0, scale=50000.0, size=n))
    btc_price_cdf  = lambda x: np.array(gamma.cdf(x, a=1.5, loc=0.0, scale=50000.0))

    eth_price = np.linspace(0.0, 20000.0, n_samples)
    eth_price_pdf  = lambda x: np.array(gamma.pdf(x, a=2.0, loc=1000.0, scale=1600.0))
    eth_price_samp = lambda n: np.array(gamma.rvs(a=2.0, loc=1000.0, scale=1600.0, size=n))
    eth_price_cdf  = lambda x: np.array(gamma.cdf(x, a=2.0, loc=1000.0, scale=1600.0)) 

    algo_price = np.linspace(0.0, 50.0, n_samples)
    algo_price_pdf  = lambda x: np.array(gamma.pdf(x, a=0.4, loc=0.0, scale=12.0))
    algo_price_samp = lambda n: np.array(gamma.rvs(a=0.4, loc=0.0, scale=12.0, size=n))
    algo_price_cdf  = lambda x: np.array(gamma.cdf(x, a=0.4, loc=0.0, scale=12.0)) 

    ada_price = np.linspace(0.0, 10.0, n_samples)
    ada_price_pdf  = lambda x: np.array(gamma.pdf(x, a=1.5, loc=0.0, scale=1.5))
    ada_price_samp = lambda n: np.array(gamma.rvs(a=1.5, loc=0.0, scale=1.5, size=n))
    ada_price_cdf  = lambda x: np.array(gamma.cdf(x, a=1.5, loc=0.0, scale=1.5)) 

    sol_price = np.linspace(0.0, 4000.0, n_samples)
    sol_price_pdf  = lambda x: np.array(gamma.pdf(x, a=0.22, loc=0.0, scale=1500.0))
    sol_price_samp = lambda n: np.array(gamma.rvs(a=0.22, loc=0.0, scale=1500.0, size=n))
    sol_price_cdf  = lambda x: np.array(gamma.cdf(x, a=0.22, loc=0.0, scale=1500.0)) 

    #fig, (ax1, ax2) = plt.subplots(1,2,sharex=True,tight_layout=True)
    #ax1.plot(sol_price, sol_price_pdf(sol_price))
    #ax1.grid()
    #ax2.plot(sol_price, sol_price_cdf(sol_price))
    #ax2.grid()
    #plt.show()
    #exit()

    usdc_price = np.linspace(0.0, 24.0, n_samples)
    usdc_return_pdf = lambda x: np.array(norm.pdf(x, 9.0, 1.5))
    usdc_return_samp = lambda n: np.array(norm.rvs(9.0, 1.5, size=n))

    token_set = TokenSet(
        token_list = [
            Token(
                name                = "USDC",
                current_price       = 1.0,
                weight              = 0.20,
                return_distribution = ReturnDistribution(labels=np.array(usdc_price), pdf_function=usdc_return_pdf, pdf_sampler=usdc_return_samp),
            ),
            Token(
                name               = "BTC",
                current_price      = 57000.0,
                weight             = 0.05,
                price_distribution = PriceDistribution(labels=np.array(btc_price), pdf_function=btc_price_pdf, pdf_sampler=btc_price_samp),
            ),
            Token(
                name               = "ETH",
                current_price      = 2900.0,
                weight             = 0.45,
                price_distribution = PriceDistribution(labels=np.array(eth_price), pdf_function=eth_price_pdf, pdf_sampler=eth_price_samp),
            ),
            Token(
                name               = "ALGO",
                current_price      = 1.39,
                weight             = 0.03,
                price_distribution = PriceDistribution(labels=np.array(algo_price), pdf_function=algo_price_pdf, pdf_sampler=algo_price_samp),
            ),
            Token(
                name               = "ADA",
                current_price      = 1.35,
                weight             = 0.07,
                price_distribution = PriceDistribution(labels=np.array(ada_price), pdf_function=ada_price_pdf, pdf_sampler=ada_price_samp),
            ),
            Token(
                name               = "SOL",
                current_price      = 45.0,
                weight             = 0.20,
                price_distribution = PriceDistribution(labels=np.array(sol_price), pdf_function=sol_price_pdf, pdf_sampler=sol_price_samp),
            )
        ]
    )
    token_dict = {}
    token_dict["USDC"] = [0,0]
    token_dict["BTC"]  = [0,1]
    token_dict["ETH"]  = [1,0]
    token_dict["ALGO"]  = [1,1]
    token_dict["ADA"]  = [2,0]
    token_dict["SOL"]  = [2,1]
    #token_set.plot_return_distributions_tiled(token_dict, 3, 2, -100.0, 1000.0)
    #token_set.plot_return_distributions_overlayed(xmin=-100.0, xmax=500.0, ymin=1.0e-8, ymax=None)
    token_set.plot_joint_return_distribution()   
 


