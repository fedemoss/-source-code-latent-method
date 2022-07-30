import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

class latent_ideology:
  """
  Method for applying the correspondence analysis method for the purpose of calculating 
  an 'ideology score' as stated in [1][2].

  [1] J. Flamino, A. Galezzi, S. Feldman, M. W. Macy, B. Cross, Z. Zhou, M. Serano, A. Bovet, H. A. Makse, and B. K. Szymanski,
  'Shifting polarization and twitter news influencers between two us presidential elections', 
  arXiv preprint arXiv:2111.02505 (2021).

  [2] Max Falkenberg, Alessandro Galeazzi, Maddalena Torricelli, Niccolo Di Marco, Francesca Larosa, Madalina Sas, Amin Mekacher, 
  Warren Pearce, Fabiana Zollo, Walter Quattrociocchi, Andrea Baronchelli,
  'Growing polarisation around climate change on social media',
  https://doi.org/10.48550/arXiv.2112.12137 (2021).

  """

  def __init__(self, df):
    self.df = df

  #from dataframe, return filtered pandas adjacency matrix 
  def make_adjacency(self, m = None ,n=2, targets = 'target', sources= 'source', weight = False, weight_name = 'weight',filtered_df = False, detailed_target_list = False):
    """
    Create weighted adjacency matrix from unfiltered -optionally- pandas dataframe input.
    The input dataframe consist of interactions between a target and a source
    ## Parameters:
    -  **m** : int (default = None). 
            Number of sources to consider (default = consider all sources in the dataset)
    -  **n** : int (default = 2). 
            Number of distinct sources interacting with each target 
    -  **targets** : str (default = 'target'). 
            Name of the column of the targets
    -  **sources** : str (default = 'source'). 
            Name of the column of the sources
    -  **weight** : bool (default = False) -if False, time of execution takes longer-
            Does your data have a weight column indicating how many times a target had interacted with each source?
            (if True, specify the column name in the next parameter).
    -  **weight_name** : str (default = False)
            if weight==True, specify the name of the column associated with the weight of the interaction target/source.
    -  **filtered_df** : bool (default = 'weight').
            Returns a filtered (given thresholds m,n) dataframe, similar to the input one.
            It also adds a column associated with the weight of the connection beetween a 
            target and a source 
    - **detailed_target_list** : bool (default = False) -if True, time of execution takes longer-
            returns another dataframe indicating which sources had the targets interacted with.
    """
    df = self.df
    df['target'] = df[targets]
    df['source'] = df[sources]
    if weight==True:
      df['weight'] = df[weight_name]
    if m==None:
      m = len(df.target) 

    #Threshold 1: number of distinct sources interacting with each target
    groups_dict_target = df[['target','source']].groupby(by=['target']).groups #dict
    keys_list = list(groups_dict_target.keys()) #users (keys)
    sources = []
    lengths = []
    if detailed_target_list == True:  
      for key in keys_list:
        source_list_index = list(groups_dict_target[key]) #sources list for each target 
        source_asocciated = []
        for s in source_list_index:
          source_asocciated.append(df.source[s])
        sources.append(list(set(source_asocciated)))
        lengths.append(len(list(set(source_asocciated)))) #list of sources lenghts
      data = {'target':keys_list, 'sources_associated':sources, 'total_retuits':lengths}#new df
      df3 = pd.DataFrame(data).sort_values(by=['total_retuits'], ascending=False).reset_index(drop=True)
      df_targets_associated = df3.query("total_retuits >= @n")
      targets_threshold_1 = list(df_targets_associated['target'])
      df_filtered_th1 = df.query('(target == @targets_threshold_1)')
    else:
      for key in keys_list:
        source_list_index = list(groups_dict_target[key]) #sources list for each target
        lengths.append(len(source_list_index))
      data = {'target':keys_list, 'total_retuits':lengths}#new df
      df3 = pd.DataFrame(data).sort_values(by=['total_retuits'], ascending=False).reset_index(drop=True)
      targets_threshold_1 = list(df3.query("total_retuits >= @n ")['target'])
      df_filtered_th1 = df.query('(target == @targets_threshold_1)')
    
    #Threshold 2: number of sources
    top_sources = df_filtered_th1[['target','source']].groupby('source').count().sort_values(by = 'target', ascending = False).head(m).index
    df_filtered_th1_th2 = df_filtered_th1[df_filtered_th1.source.isin(top_sources)].copy()

    #Unique targets, sources
    source_list = list(df_filtered_th1_th2.source.unique())
    target_list = list(df_filtered_th1_th2.target.unique())

    if detailed_target_list:
      targets_info = df_targets_associated[df_targets_associated.target.isin(target_list)]

    #Weights
    if weight == False:
      #Lets add the weight of each interaction
      w = []
      df4 = pd.DataFrame()
      for s in source_list:
        df4['weight'] = df_filtered_th1_th2[df_filtered_th1_th2.source==s][['target','source']].groupby(by='target').source.count()
        df4['source'] = s
        w.append(df4)
        df4 = pd.DataFrame()
      dfw = pd.concat(w).fillna(0).sort_values('weight',ascending=False).groupby(by=['target','source']).sum().reset_index()
    else:
      dfw = df_filtered_th1_th2

    #Final matrix
    source_col = []
    for s in source_list:
      df_aux = dfw[dfw.source == s][['target','weight']].set_index('target') #column users & weights
      df_aux.columns = [s] #rename weight --> influencer asociated 
      source_col.append(df_aux) #list of targets associated with source

    final_data = pd.concat(source_col).fillna(0).groupby('target').sum()

    if filtered_df == True and detailed_target_list == False:
      return dfw, final_data
    elif filtered_df == False and detailed_target_list == False:
      return final_data
    elif filtered_df == True and detailed_target_list == True:
      return dfw, targets_info, final_data
    elif filtered_df == False and detailed_target_list == True:
      return targets_info, final_data
    

  #Use the correpondence analysis method to calculate the scores of a given adjacency matrix in the rows projection
  def calculate_scores(self, A, dimension = 1):
    """
    Normalize, standardized and use SVD to reduce the dimension of a given input matrix.
    The output is a 'score' associated with each row of the input matrix. 
    If multiple dimensions are consider, the 'score' output is a list of scores associated 
    with each dimension (each principal component considered)

    OBS: Since the scores are 'row scores', one can simply trnaspose the input matrix to calculate the
    'column scores'. In the case of an adjacency matrix:
      -  row scores == target scores
      -  column scores == sources scores

  
    This method is further discussed in [1] and [2].
    ## Parameters
    -  **A** : numpy matrix.
            Weighted adjacency matrix.
    -  **dimension** : int (default = 1).
            To how many dimensions shall the truncated SVD method reduce the input matrix A. 
            This is equivalent the number of principal components considered
            when truncating the SVD method.
    """

    P = (1/np.sum(A))*A #Nomalized natrix

    #Defining needings for standardizing
    n_col = np.shape(P)[1]
    n_row = np.shape(P)[0]
    r = np.matmul(P, np.ones((n_col,))) #rows
    c = np.matmul(np.ones((n_row,)), P) #columns
    r2 = r**(-0.5)
    c2 = c**(-0.5)
    Dr2 = np.diag(r2)
    Dc2 = np.diag(c2)
    r_t = np.array([r]).transpose()
    c_new = np.array([c])

    #Standardized residuals
    S = np.matmul(np.matmul(Dr2, P - np.matmul(r_t,c_new)),Dc2)

    if dimension > 1:
      #Truncated SVD
      U, sig, Vt = randomized_svd(S, n_components=dimension, n_iter=5, random_state=None)
      X_dim1 = np.matmul(Dr2,U) #scores matrix
      scores = []
      for i in range(dimension):
        #scaling betweeen -1 and 1 each dimension
        scores.append((-1 + 2 * (X_dim1[:,i]-np.min(X_dim1[:,i]))/np.ptp(X_dim1[:,i]))) #scaled
    else:
      #Truncated SVD
      U, sig, Vt = randomized_svd(S, n_components=1, n_iter=5, random_state=None)
      
      #scores
      X_dim1 = np.matmul(Dr2,U) #scores matrix
      scores = (-1 + 2 * (X_dim1-np.min(X_dim1))/np.ptp(X_dim1)) #scaled
    
    return scores


  #Compute the scores for rows and columns using the built-it correspondence analysis method. 
  #Here, sources scores are calculated transposing the adjacency matrix
  def apply_simplified_method(self, df_adjacency):
    """
    Apply the correspondence analysis method to calculate the row scores given an adjacency matrix.
    The column scores (or the score of the sources) are calculated by transposing the adjacency matrix
    and imposing the exact same treatment as with the original non-transposed adjacency matrix. 

    ## Parameters
   -   **df_adjacency** : pandas dataframe. 
                  Weighted adjacency matrix in the shape of a pandas dataframe.
                  This could be the output from the make_adjacency() funciton.

    """

    A = df_adjacency.to_numpy(dtype = int) #for row scores
    B = df_adjacency.T.to_numpy(dtype = int) #for column scores
    row_scores = self.calculate_scores(A)
    col_scores = self.calculate_scores(B)

    #DataFrame of targets (rows) scores
    scores_list = [float(l) for l in row_scores]
    data_metodo = {'target':df_adjacency.index,'score':scores_list}
    df_scores_target = pd.DataFrame(data_metodo)

    #DataFrame of sources (columns) scores
    scores_list = [float(l) for l in col_scores]
    data_metodo = {'source':df_adjacency.columns,'score':scores_list}
    df_scores_sources = pd.DataFrame(data_metodo)

    return df_scores_target, df_scores_sources

#Compute the scores for rows and columns using the built-it correspondence analysis method as stated in the bibliography
  def apply_method(self,  m = None ,n=2, targets = 'target', sources= 'source', weight = False, weight_name = 'weight'):
    """
    Apply the correspondence analysis method to calculate the row scores given a source-target interaction dataframe.
    The column scores (or the score of the sources) correspond to the mean othe target scores
    associated with a given source. Just like [1][2].

    This function transform a dataframe of interactions into an adjacency matrix. This proccess takes
    into consideration some filtering. 
    This can be further studied by looking into the built-in function 'make_adjacency()'

    ## Parameters
    -  **m** : int (default = None). 
            Number of sources to consider (default = consider all sources in the dataset)
    -  **n** : int (default = 2). 
            Number of distinct sources interacting with each target 
    -  **targets** : str (default = 'target'). 
            Name of the column of the targets
    -  **sources** : str (default = 'source'). 
            Name of the column of the sources
    -  **weight** : bool (default = False) -if False, time of execution takes longer-
            Does your data have a weight column indicating how many times a target had interacted with each source?
            (if True, specify the column name in the next parameter).
    -  **weight_name** : str (default = False)
            if weight==True, specify the name of the column associated with the weight of the interaction target/source.

    """

    #Adjacency matrix & filtering
    df_filtered, df_adjacency = self.make_adjacency(m,n,targets=targets, sources=sources, filtered_df=True, weight=weight, weight_name=weight_name)

    #DataFrame of targets (rows) scores
    A = df_adjacency.to_numpy(dtype = int) #for row scores
    row_scores = self.calculate_scores(A)
    scores_list = [float(l) for l in row_scores]
    data_metodo = {'target':df_adjacency.index,'score':scores_list}
    df_scores_target = pd.DataFrame(data_metodo)

    #DataFrame of sources (columns) scores
    df_final = df_filtered.set_index('target').join(df_scores_target.set_index('target'))
    df_final['target'] = df_final.index
    df_final = df_final.reset_index(drop=True).copy()

    groups_dict = df_final[['source','score']].set_index('score').groupby(by=['source']).groups
    keys_list = list(groups_dict.keys()) #los influencers (son keys)
    mean_scores = []
    for key in keys_list:
      score_list = list(groups_dict[key]) #lista de scores
      mean_scores.append(np.mean(score_list))

    data_new = {'source':[str(key) for key in keys_list], 'score':mean_scores} #Create dataframe
    df_scores_source = pd.DataFrame(data_new).sort_values(by=['score'], ascending=False).reset_index(drop=True)

    return df_scores_target, df_scores_source
