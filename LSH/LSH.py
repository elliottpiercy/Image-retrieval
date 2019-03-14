import numpy as np
from sklearn.neighbors import LSHForest
import itertools

class LSH():
    
    def __init__(self,train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test
        

    # Return grid search variable generator based on parameters.
    def get_generator(self, parameters):

        keys = parameters.keys()
        values = (parameters[key] for key in keys)
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        for combo in combinations:
            yield combo
                    
    
    # Optimise the model using train and validation sets. The number of points used is outlined in the config file.
    def optimise(self, num_train_points, num_val_points, parameters):

        max_accuracy = -1
        optimal_estimators = -1
        optimal_n_neighbours = -1


        for item in self.get_generator(parameters):
            
            LSHf = LSHForest(random_state=42, n_estimators = item['n_est'], n_neighbors = item['n_neigh'])
            LSHf.fit(self.train.images[:num_train_points])
            distances, indices = LSHf.kneighbors(self.validation.images[:num_val_points], n_neighbors = 5)

            accuracy, positions = self.model_accuracy(indices, is_optimising= True)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                optimal_estimators = item['n_est']
                optimal_n_neighbours = item['n_neigh']

#         print(optimal_n_neighbours_predict)
        return max_accuracy, optimal_estimators, optimal_n_neighbours
    
    
    
    # Find the models accuracy
    def model_accuracy(self, predicted_labels, is_optimising = False):

        correct = 0 
        wrong = 0

        # Dictionary showing the distribution of correct and incorrect predictions
        positions = {}
        for prediction in range(len(predicted_labels[0])):
            positions[prediction] = 0
        
        
        for row in range(predicted_labels.shape[0]):
            true_label = self.validation.labels[row]
            for item in range(len(predicted_labels[0])):
                if is_optimising:
                    pred_label = self.train.labels[int(predicted_labels[row,item])]
                else:
                    pred_label = self.validation.labels[int(predicted_labels[row,item])]
                if pred_label == true_label:
                    correct +=1
                    positions[item] += 1
               
                else:
                    wrong +=1
                    
        accuracy = (correct/(correct+wrong)) * 100

        print("Correct:",correct,". Wrong:",wrong,".")
        print("Percentage class error:",accuracy)
        
        return accuracy, positions
    
    
    def plot_bar_chart(self, positions, num_of_samples):
    
        # Creates vector from dictionary with density of relevant predictions.
        hist = []
        pred_loc = []
        for i in positions:
            hist = np.append(hist,positions[i])
            pred_loc = np.append(pred_loc,i+1)
        
        hist = num_of_samples -hist
        

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        x_pos = [i for i, _ in enumerate(pred_loc)]

        plt.bar(x_pos, hist, color='green')
        plt.xlabel("Ranking")
        plt.ylabel("Number of incorrect predictions")
        plt.title("Incorrect ranked predictions for each ranking")

        plt.xticks(x_pos, pred_loc)

        plt.show()
        
        
    # Fit model using the optimised parameters
    def fit_model(self, data, n_estimators, n_neighbours):
        
        LSHf = LSHForest(random_state=42, n_estimators = n_estimators, n_neighbors = n_neighbours)
        LSHf.fit(data)
        return LSHf


    # Predict using the optimal model
    def predict(self, model, data, neighbours):
        
        distances, indices = model.kneighbors(data, n_neighbors = neighbours+1)
        return np.delete(indices,0,axis=1)