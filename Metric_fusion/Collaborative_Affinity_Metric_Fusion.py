import numpy as np

class collaborative_affinity_metric_fusion():

    def __init__(self):
        pass

    # Creates global affinity matrix by computing the euclidean distance between image feature vectors and applying non-linearity
    # Equation 3
    def affinity_matrix(self, features):
        W = np.array([])
        control_constant = self.get_median(features)


        from sklearn.metrics.pairwise import euclidean_distances
        for feature_a in features:
            for feature_b in features:

                euclidean_dist = euclidean_distances(np.atleast_2d(feature_a),np.atleast_2d(feature_b))
                W = np.append(W,np.exp(euclidean_dist / control_constant))

        return np.reshape(W,(len(features),len(features)))



    # Calculates the median of the distances between feature vectors. Used as a control constant.
    def get_median(self, features):
        distances = np.array([])
        from sklearn.metrics.pairwise import euclidean_distances
        for i in features:
            for j in features:
                distances = np.append(distances,euclidean_distances(np.atleast_2d(i),np.atleast_2d(j)))

        return np.median(distances)




    # Creates a local affinity matrix. If the neighbouring nodes are a minimal neighbour then keep the value. Else = 0.
    # Equation 5.
    def local_affinity_matrix(self, W, neighbours):
        # Iterate over rows
        for row in range(len(W)):
            # Find closest N neighbours
            min_list = np.argsort(W[row])[:neighbours]
            # Find set difference and set larger neighbours to 0
            invalid_neighbours = list(set(np.arange(len(W[row])))-(set(min_list)))
            W[row][invalid_neighbours] = 0
        return W

    

    # Normalise W along each row. 
    def normalise(self, data):
        for row in range(len(data)):
            data[row] = data[row] / np.sum(data[row])
        return np.reshape(data,(1,data.shape[0],data.shape[1]))
    
    

    # Normalise W to give us the status matrix. Global graph. Equation 4.
    def status_matrix_creation(self, W):
        return self.normalise(W)


    # Normalise W_ to give us the status matrix. Local graph. Equation 6.
    def kernel_matrix_creation(self, W_):
        return self.normalise(W_)



    # Creates the status and kernel matrices based on the extracted features and the number of local neighbours
    def create_graphs(self, features, neighbours):
        status_matrix = np.zeros((1,len(features),len(features)))
        kernel_matrix = np.zeros((1,len(features),len(features)))


        W = self.affinity_matrix(features)
        s_matrix = self.status_matrix_creation(W)
        status_matrix = np.concatenate((status_matrix,s_matrix),axis=0)

        from copy import deepcopy
        W_ = self.local_affinity_matrix(deepcopy(W),neighbours)
        k_matrix = self.kernel_matrix_creation(W_)
        kernel_matrix = np.concatenate((kernel_matrix,k_matrix),axis=0)

        return np.delete(status_matrix,0,axis=0), np.delete(kernel_matrix,0,axis=0)
    
    # Find the diffusion indexs due to the fact k!=m. Randomly shuffle until k!=m.
    def status_diffusion_index(self, graphs):
        M = np.arange(graphs)
        np.random.shuffle(M)

        while True:
            if (M != np.arange(graphs)).all():
                break
            else:
                np.random.shuffle(M)
        return M
    

    # Applys cross diffusion to the status matrices and kernel matrices. This merges the features gathered from multiple 
    # algorithms into one matrix that can be queried for image similarity. Equation 7.
    def cross_diffusion(self, status_matrix, kernel_matrix):
        T = 20
        eta = 1
        k = self.status_diffusion_index(status_matrix.shape[0])
        M = status_matrix.shape[0]
        
        # Iterate over feature graphs
        for graph in range(len(k)):
            for t in range(T):
                # Application of equation 7.
                status_matrix[graph] = kernel_matrix[graph] * ((1/M-1) * np.sum(status_matrix[k[graph]])) * kernel_matrix[graph].T + eta * np.identity(kernel_matrix.shape[1])

        return np.mean(status_matrix,axis=0)


    # Merges graphs into ones array to be used in the cross diffusion step.
    def merge_graph(self, graph_1, graph_2):
        return np.concatenate((graph_1, graph_2), axis=0)
    
    
    # Find the n closest images using W_FAM matrix.
    def image_similarity(self, W_FAM, neighbours):
        image_locations = np.array([])
        
        for row in W_FAM:
            # Locate min n neighbours ignoring the 1st position due to self similarity.
            min_distance = np.argsort(row)[::-1][1 :neighbours+1]
            image_locations = np.append(image_locations, min_distance)
        
        return np.reshape(image_locations,(W_FAM.shape[0],neighbours))