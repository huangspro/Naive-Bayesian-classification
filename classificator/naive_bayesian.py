class NaiveBayesianClassificator:
    def __init__(self, dataset, LS = True):
        self.p_c = {"SEKER": 0, "BARBUNYA": 0, "BOMBAY": 0, "CALI": 0, "DERMASON": 0, "HOROZ": 0, "SIRA": 0}
        self.LS = LS
        self.dataset = dataset
        
    def fit(self):
        for i in self.dataset.train_set:
            self.p_c[self.dataset.get_in_train(i)[1]] += 1
        for k, v in self.p_c.items():
            self.p_c[k] = v / self.dataset.length()[1]
    
    # calcullate P(xi|c) in train_set, X = (x1, x2, ...), c is the class
    # IMPORTANT, the order of features in X must be the same as the one in dataset 
    def get_p_xi_c(self, xi, xi_index, c):
        c_number = 0
        xi_number = 0
        for i in self.dataset.train_set:
            if self.dataset.get_in_train(i)[1] == c:
                c_number += 1
                if self.dataset.get_in_train(i)[0][xi_index] == xi:
                    xi_number += 1
        
        return xi_number/c_number
        
    def predict(self, inputs):
        result = {"SEKER": 0, "BARBUNYA": 0, "BOMBAY": 0, "CALI": 0, "DERMASON": 0, "HOROZ": 0, "SIRA": 0}
        for c in result.keys():
            # calculate P(X|c)
            p_x_c = 1
            for xi_index, xi in enumerate(inputs):
                p_x_c *= self.get_p_xi_c(xi, xi_index, c)
            result[c] = p_x_c * self.p_c[c]
        
        # get class name
        pred_prob = 0  # the probability is not the real probability, beacause we don not divide it by P(X)
        pred_class = 0
        for k,v in result.items():
            if v > pred_prob:
                pred_prob = v
                pred_class = k 
        return (pred_class, pred_prob)
