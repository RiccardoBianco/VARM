import numpy as np
from scipy.spatial.distance import cdist


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be -1 if there is no database descriptor
    with an SSD < lambda * min(SSD). No elements of matches will be equal except for the -1 elements.
    """
    pass
    # TODO: Your code here
    # il query_descriptors è un descrittore che rappresenta tutti i keypoints del nuovo frame (Q keypoints)
    # il database_descriptors è un descrittore che rappresenta tutti i keypoints del drame precedente (D keypoints)
    # distances è una matrice delle distanze che calcola la distanza di ogni keypoint del query (nuovo frame) da ogni keypoints del vecchio frame
    
    # Non è detto che due immagini consecutive abbiano lo stesso numero keypoints --> per questo le identifico come MxQ e MxD
    # Tuttavia nel nostro caso noi consideriamo un numero di keypoints uguale e la dimensione del patch uguale, quindi la dimensione del 
    # descriptor è dxk (d dimenisione del patch in pixel, k numero di keypoints)
    
    distances = cdist(query_descriptors.T, database_descriptors.T, metric='euclidean') # dimensione QxD
    # dove distances[i, j] rappresenta la distanza euclidea tra il i-esimo descrittore di query_descriptors e il j-esimo descrittore di database_descriptors.
    
    # distances è una matrice QxD --> voglio prendere il keypoint del nuovo frame più vicino tra tutti i keypoints del vecchio frame
    # creo un array di dimensione Q (matches) in cui associo per ogni keypoint del nuovo frame, il keypoint del vecchio frame che ha distanza minima
    matches = np.argmin(distances, axis=1) # trova, per ogni riga (cioè per ogni descrittore query), l'indice della colonna con la distanza minima.
    
    distances = distances[np.arange(matches.shape[0]), matches] # per ogni match keypoint del vecchio frame e keypoint del nuovo, salvo la distanza tra questi
    # distances ora è un array di dimensione Q che contiene le distanze minime per ciascun descrittore query
    
    min_distance = distances.min()
    adaptive_threshold = match_lambda * min_distance
    
    matches[distances >= adaptive_threshold] = -1
    
    
    
    unique_matches = -np.ones_like(matches) # inizializzato a -1 per conservare i match non validi
    _, unique_match_idxs = np.unique(matches, return_index=True) # salvo l'indice corrisondente al primo match --> nel caso di più match
    unique_matches[unique_match_idxs] = matches[unique_match_idxs] # mi assicuro che in unique_matches compaia solo il primo match
    
    return unique_matches

