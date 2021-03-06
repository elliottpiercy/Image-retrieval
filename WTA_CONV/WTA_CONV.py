import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

class WTA_CONV:

    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

        
    
    def encoder(self, inputs, reuse):
        
        with tf.variable_scope("encoder", reuse = reuse):
    
    
            BN0 = tf.layers.batch_normalization(inputs)
            
            conv1 = tf.layers.conv2d(BN0, 16, (3,3), padding='same', activation=tf.nn.relu)

            # Now 28x28x16
            maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
            
            BN1 = tf.layers.batch_normalization(maxpool1)
            # Now 14x14x16
            conv2 = tf.layers.conv2d(BN1, 8, (3,3), padding='same', activation=tf.nn.relu)
            # Now 14x14x8
            maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
            
            BN2 = tf.layers.batch_normalization(maxpool2)
            # Now 7x7x8
            conv3 = tf.layers.conv2d(BN2, 8, (3,3), padding='same', activation=tf.nn.relu)
            # Now 7x7x8
            encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
            
        return encoded
    
    
    def decoder(self, encoded, reuse):
        
        with tf.variable_scope("decoder", reuse = reuse):
            
            ### Decoder
            BN3 = tf.layers.batch_normalization(encoded)

            upsample1 = tf.image.resize_nearest_neighbor(BN3, (7,7))
            # Now 7x7x8
            conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
            # Now 7x7x8
            upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
            
            BN4 = tf.layers.batch_normalization(upsample2)
            # Now 14x14x8
            conv5 = tf.layers.conv2d(BN4, 8, (3,3), padding='same', activation=tf.nn.relu)
            # Now 14x14x8
            upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
            
            BN4 = tf.layers.batch_normalization(upsample3)
            # Now 28x28x8
            conv6 = tf.layers.conv2d(BN4, 16, (3,3), padding='same', activation=tf.nn.relu)
            # Now 28x28x16

            logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
            #Now 28x28x1
            
            decoded = tf.nn.sigmoid(logits, name='decoded')
            
            return decoded
    
    
    # Return complete model
    def model(self,inputs,config,reuse):
        
  
        with tf.variable_scope("whole_model", reuse=reuse):
        
            ### Encoder
            
            encoded = self.encoder(inputs, reuse)
            state, _ = self._spatial_sparsity(encoded)
            decoded = self.decoder(state, reuse)
           
        
        return decoded,encoded


    def _spatial_sparsity(self, h):
        shape = tf.shape(h)
        n = shape[0]
        c = shape[3]

        h_t = tf.transpose(h, [0, 3, 1, 2]) # n, c, h, w
        h_r = tf.reshape(h_t, tf.stack([n, c, -1])) # n, c, h*w

        th, _ = tf.nn.top_k(h_r, 1) # n, c, 1
        th_r = tf.reshape(th, tf.stack([n, 1, 1, c])) # n, 1, 1, c
        drop = tf.where(h < th_r, 
          tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))

        # spatially dropped & winner
        return h*drop, tf.reshape(th, tf.stack([n, c])) # n, c
    
    
    def _lifetime_sparsity(self, h, winner, rate):
        shape = tf.shape(winner)
        n = shape[0]
        c = shape[1]
        k = tf.cast(rate * tf.cast(n, tf.float32), tf.int32)

        winner = tf.transpose(winner) # c, n
        th_k, _ = tf.nn.top_k(winner, k) # c, k

        shape_t = tf.stack([c, n])
        drop = tf.where(winner < th_k[:,k-1:k], # c, n
          tf.zeros(shape_t, tf.float32), tf.ones(shape_t, tf.float32))
        drop = tf.transpose(drop) # n, c
        return h * tf.reshape(drop, tf.stack([n, 1, 1, c]))

    
    
    def loss(self, targets, model, lifetime_sparsity):
        _, encoded = model
        state, winner = self._spatial_sparsity(encoded)
        state = self._lifetime_sparsity(state, winner, lifetime_sparsity)
        output = self.decoder(state,False)

        return tf.reduce_sum(tf.losses.mean_squared_error(labels=targets, predictions =  output))

    
    
    
    # Return model embedding
    def get_embedding(self, model):
    
        _, embedding = model
        return embedding


        
    # Flatten embeddings into vectors
    def flatten_embeddings(self,embedding):
        flatten_embeddings = []

        for i in range(embedding.shape[0]):
            flatten_embeddings = np.append(flatten_embeddings, np.ravel(embedding[i]))

        reshape_size = embedding.shape[1] * embedding.shape[2] * embedding.shape[3]
        flatten_embeddings = np.reshape(flatten_embeddings,(embedding.shape[0],reshape_size))

        # Return flattened embeddings
        return flatten_embeddings
    
    
    
    
    # Fit KDTree with embeddings
    def fit_KDTree(self, embeddings):
        embeddings = self.flatten_embeddings(embeddings)
        
        from sklearn.neighbors import KDTree   
        tree = KDTree(embeddings)
        return tree           
        
              
            
            
    # Return nearest neighbours of KDtree. Performance appears to be exactly the same as KNN
    def tree_nearest_neighbours(self, model, embedding, neighbours):
        embedding = np.atleast_2d(np.ndarray.flatten(embedding))
        
        distances, indices = model.query(embedding, neighbours+1)
        # First neighbours is itself so ignore it.
        indices = indices[0][1:]
        
        return indices
    
    
    
        
    # Fit KNN classifier with embeddings and return the classfier
    def fit_KNN(self, embeddings, neighbours, p=2):
        embeddings = self.flatten_embeddings(embeddings)
        
        from sklearn.neighbors import KNeighborsClassifier
        # neighbours = neighbours + 1 as the first neighbours will be itself
        classifier = KNeighborsClassifier(n_neighbors=neighbours+1,p=p)
        classifier.fit(embeddings, np.arange(embeddings.shape[0]))
        return classifier
    
    
    
    
    # Find nearest neighbours from parsed classifier and embedding
    def K_nearest_neighbours(self, model, embedding):
        embedding = np.atleast_2d(np.ndarray.flatten(embedding))
        
        distances, indices = model.kneighbors(embedding)
        # First neighbours is itself so ignore it.
        indices = indices[0][1:]
        
        return indices
    
    
    
    
    # Print the model accuracy given the predicted labels
    def model_accuracy(self, predicted_labels, num_of_samples, num_predictions):

        correct = 0 
        wrong = 0

        # Dictionary showing the distribution of correct and incorrect predictions
        positions = {}
        for prediction in range(num_predictions):
            positions[prediction] = 0
        
        
        for row in range(num_of_samples):
            true_label = self.validation.labels[row]
            for item in range(num_predictions):
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
    
    
    
    
    # Plot bar chart showing the accuracy of ranked predictions
    def plot_bar_chart(self, positions, num_samples):
    
        # Creates vector from dictionary with density of relevant predictions.
        hist = []
        pred_loc = []
        for i in positions:
            hist = np.append(hist,positions[i])
            pred_loc = np.append(pred_loc,i+1)
        
        hist = num_samples -hist
        

        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        x_pos = [i for i, _ in enumerate(pred_loc)]

        plt.bar(x_pos, hist, color='green')
        plt.xlabel("Ranking")
        plt.ylabel("Number of correct predictions")
        plt.title("Incorrect ranked predictions for each ranking")

        plt.xticks(x_pos, pred_loc)

        plt.show()
    
    

 