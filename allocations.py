import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm

class ReturnDistribution(object):
    def __init__(self, current_price=None, projected_price_distribution=None, labels=None, likelihoods=None):
        assert (current_price is not None and projected_price_distribution is not None) ^ (labels is not None and likelihoods is not None)
        self.current_price = current_price
        if current_price is not None:
            self.labels = np.zeros(projected_price_distribution.num_bins, dtype=np.float)
            self.likelihoods = np.copy(projected_price_distribution.likelihoods)
            self.num_bins = projected_price_distribution.num_bins
            for i in range(0,self.labels.shape[0]):
                self.labels[i] = 100.0 * (projected_price_distribution.labels[i]/self.current_price - 1.0)
        else:
            self.labels = labels
            self.likelihoods = likelihoods
    
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
    def __init__(self, labels, likelihoods):
        self.likelihoods = likelihoods
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
        for i, token in enumerate(self.token_list):
            self.name_index_map[token.name] = token

    def token(self, name):
        return self.name_index_map[name]

    def plot_joint_return_distribution(self):
        assert False, "TODO"
 
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
            ax[index[0]*num_cols + index[1]].plot(token.return_distribution.labels, token.return_distribution.likelihoods, label=token.name)
            ax[index[0]*num_cols + index[1]].legend()
        plt.show()

#Use case example
if __name__=="__main__":

    btc_price = np.linspace(0.0, 450000.0, 128)
    btc_likelihoods = gamma.pdf(btc_price, a=2.5, loc=0.0, scale=40000.0)

    eth_price = np.linspace(0.0, 20000.0, 128)
    eth_likelihoods = gamma.pdf(eth_price, a=2.5, loc=1000.0, scale=1500.0)

    usdc_price = np.linspace(0.0, 24.0, 128)
    usdc_likelihoods = norm.pdf(usdc_price, 9.0, 1.5)

    token_set = TokenSet(
        token_list = [
            Token(
                name                = "USDC",
                current_price       = 1.0,
                weight              = 0.6,
                return_distribution = ReturnDistribution(labels=np.array(usdc_price), likelihoods=np.array(usdc_likelihoods)),
            ),
            Token(
                name               = "BTC",
                current_price      = 57000.0,
                weight             = 0.10,
                price_distribution = PriceDistribution(labels=np.array(btc_price), likelihoods = np.array(btc_likelihoods)),
            ),
            Token(
                name               = "ETH",
                current_price      = 2900.0,
                weight             = 0.3,
                price_distribution = PriceDistribution(labels=np.array(eth_price), likelihoods=np.array(eth_likelihoods)),
            )
        ]
    )
    token_dict = {}
    token_dict["USDC"] = [0,0]
    token_dict["BTC"]  = [0,1]
    token_dict["ETH"]  = [0,2]
    token_set.plot_return_distributions_tiled(token_dict, 1, 3, 0.0, 400.0)
    token_set.plot_return_distributions_overlayed(xmin=-100.0, xmax=400.0, ymin=1.0e-8, ymax=None)
    token_set.plot_joint_return_distribution()   
 


