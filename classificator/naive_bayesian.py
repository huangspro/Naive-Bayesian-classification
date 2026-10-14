class NaiveBayesianClassificator:
    def __init__(self, dataset, LS = True):
        self.p_c = {"SEKER": 0, "BARBUNYA": 0, "BOMBAY": 0, "CALI": 0, "DERMASON": 0, "HOROZ": 0, "SIRA": 0}
        self.p_xi_c = {}
        self.xi = []
        self.LS = LS
        self.dataset = dataset
        
    def fit(self):
        # count features
        for i in self.dataset.train_set:
            tem = self.dataset.get_in_train(i)
            self.p_c[tem[1]] += 1
            for index, feature in enumerate(tem[0]):
                if (tem[1], index, feature) in self.p_xi_c:
                    self.p_xi_c[(tem[1], index, feature)] += 1
                else:
                    self.p_xi_c[(tem[1], index, feature)] = 0
        
        # get each feature possible values
        self.xi = [[] for i in range(16)]
        for k, v in self.p_xi_c.items(): 
            if k[2] not in self.xi[k[1]]:
                self.xi[k[1]].append(k[2])
        self.xi = [len(i) for i in self.xi]
        
        # calculate probability
        # self.p_xi_c: {(classname, feature_index, feature_value)}
        for k, v in self.p_xi_c.items(): 
            if self.LS:
                self.p_xi_c[k] = (self.p_xi_c[k] + 1) / (self.p_c[k[0]] + self.xi[k[1]])  # add sth 
            else:
                self.p_xi_c[k] = self.p_xi_c[k] / self.p_c[k[0]]
        for k, v in self.p_c.items():
            self.p_c[k] = v / self.dataset.length()[1]
            
        
        print(self.xi)
        
    def predict(self, inputs):
        result = {"SEKER": 1, "BARBUNYA": 1, "BOMBAY": 1, "CALI": 1, "DERMASON": 1, "HOROZ": 1, "SIRA": 1}
        # calculate P(X|c)
        for c in result.keys():
            for index, feature in enumerate(inputs):
                if (c, index, feature) in self.p_xi_c:
                    result[c] *= self.p_xi_c[(c, index, feature)]
                else:
                    result[c] *= 0
            result[c] *= self.p_c[c]
                
        # get class name
        pred_prob = 0  # the probability is not the real probability, beacause we don not divide it by P(X)
        pred_class = 0
        for k,v in result.items():
            if v > pred_prob:
                pred_prob = v
                pred_class = k 
        return (pred_class, pred_prob)
