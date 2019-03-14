import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

class DCAE:

    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test


    # Train deep convolutional autoencoder.
    def optimise(self, config):
      
        tf.reset_default_graph()

        inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
        targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

        ### Encoder
        conv1 = tf.layers.conv2d(inputs_, 16, (3,3), padding='same', activation=tf.nn.relu)
        # Now 28x28x16
        maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
        # Now 14x14x16
        conv2 = tf.layers.conv2d(maxpool1, 8, (3,3), padding='same', activation=tf.nn.relu)
        # Now 14x14x8
        maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
        # Now 7x7x8
        conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
        # Now 7x7x8
        encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
        # Now 4x4x8



        ### Decoder
        upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
        # Now 7x7x8
        conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
        # Now 7x7x8
        upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
        # Now 14x14x8
        conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
        # Now 14x14x8
        upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
        # Now 28x28x8
        conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
        # Now 28x28x16

        logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
        #Now 28x28x1

        decoded = tf.nn.sigmoid(logits, name='decoded')

        # MSE loss fn
        loss = tf.losses.mean_squared_error(labels=targets_, predictions = decoded)
        cost = tf.reduce_mean(loss)
        # Adam learning rate
        opt = tf.train.AdamOptimizer(config.learning_rate).minimize(cost)

        sess = tf.Session()


        sess.run(tf.global_variables_initializer())
        for e in range(config.epochs):
            for ii in range(self.train.num_examples//config.batch_size):
                batch = self.train.next_batch(config.batch_size)
                imgs = batch[0].reshape((-1, 28, 28, 1))
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                                 targets_: imgs})

                print("Epoch: {}/{}...".format(e+1, config.epochs),
                      "Training loss: {:.4f}".format(batch_cost))
            saver = tf.train.Saver()
            
            name = config.save_name + '_epoch_'+str(e)+'.cptk'
            save_path = saver.save(sess, os.path.join(config.save_path, name))

        
        
    # Test DCAE on MNIST test data returning the encoded states
    def test(self,config):

        tf.reset_default_graph()

        inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
        targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

        ### Encoder
        conv1 = tf.layers.conv2d(inputs_, 16, (3,3), padding='same', activation=tf.nn.relu)
        # Now 28x28x16
        maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
        # Now 14x14x16
        conv2 = tf.layers.conv2d(maxpool1, 8, (3,3), padding='same', activation=tf.nn.relu)
        # Now 14x14x8
        maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
        # Now 7x7x8
        conv3 = tf.layers.conv2d(maxpool2, 8, (3,3), padding='same', activation=tf.nn.relu)
        # Now 7x7x8
        encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
        # Now 4x4x8



        sess = tf.Session()
        saver = tf.train.Saver()

        saver = saver.restore(sess, os.path.join(config.save_path,config.save_name))

        in_imgs = self.validation.images[:config.num_test_imgs]
        embeddings = sess.run(encoded, feed_dict={inputs_: in_imgs.reshape((config.num_test_imgs, 28, 28, 1))})

        # Return encoded embeddings
        return embeddings
    
    

    # Flatten embeddings into vectores
    def flatten_embeddings(self,embedding):
        flatten_embeddings = []

        for i in range(embedding.shape[0]):
            flatten_embeddings = np.append(flatten_embeddings, np.ravel(embedding[i]))

        reshape_size = embedding.shape[1] * embedding.shape[2] * embedding.shape[3]
        flatten_embeddings = np.reshape(flatten_embeddings,(embedding.shape[0],reshape_size))

        # Return flattened embeddings
        return flatten_embeddings

    
    # p = distance metric. 1 = Manhatten. 2 = Euclidean
    # Fit KNN classifier to the embeddings
    def fit_KNN(self, embeddings, neighbours, p=2):
        embeddings = self.flatten_embeddings(embeddings)
        
        from sklearn.neighbors import KNeighborsClassifier
        # neighbours = neighbours + 1 as the first neighbours will be itself
        classifier = KNeighborsClassifier(n_neighbors=neighbours+1,p=p)
        classifier.fit(embeddings, np.arange(embeddings.shape[0]))
        return classifier
    
    
    
    # Find nearest neighbours from parsed classifier and embedding
    def nearest_neighbours(self, model, embedding):
        embedding = np.atleast_2d(np.ndarray.flatten(embedding))
        
        distances, indices = model.kneighbors(embedding)
        # First neighbours is itself so ignore it.
        indices = indices[0][1:]
        
        return indices
    
    
    # Returns the accuracy and number of correct predictions for each predicted ranking.
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
    
    
    # Plots the number of correct predictions for each ranking respectively.
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
    
    

