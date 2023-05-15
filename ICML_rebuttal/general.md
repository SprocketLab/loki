We thank the reviewers for their thoughtful and thorough reviews. Many of the suggested improvements were aligned with our ongoing work to improve the paper, both empirically and theoretically. A major theme across the reviews was that our empirical evaluation could be improved by applying our method to datasets with a larger number of classes -- we agree with this, and we have been working to include these experiments. 


**New experimental results**


We ran several experiments on two new datasets: 
a 10,000 class subset of the LSHTC dataset [1] (which was specifically mentioned by reviewer xhoG) where our metric space is derived from their class graph (which includes 325,000+ classes), and
a 9,419 class subset of a Biomedical PubMed Articles dataset [2] where we derive our metric space from Euclidean distances between label embeddings using SimCSE [3]. 




We provide the results of these experiments in the tables below. For simplicity, we use the baseline found in [1.a], a 5-nearest neighbor classifier, for both of these datasets. As before, K refers to the number of observed classes, and we report results as average squared distances.


LSHTC (10,000 classes):
| K    | 5-NN       | 5-NN + Loki    |
|------|------------|----------------|
| 100  | 3485.46487 | **2707.10019** |
| 250  | 4612.20988 | **2714.12089** |
| 500  | 3610.28105 | **2373.41991** |
| 750  | 3803.32239 | **3013.06586** |
| 1000 | 5382.15325 | **2164.65027** |


PubMed (9,419 classes):
| K   | 5-NN    | 5-NN + Loki |
|-----|---------|-------------|
| 100 | 1.68581 | **1.41822** |
| 250 | 1.52380 | **1.47278** |
| 500 | 1.64563 | **1.45370** |


In these settings, Loki continues to improve over the baseline. 


**New theoretical results**




We have also significantly generalized our sample complexity result. Originally, our result in Lemma 4.1 obtains a sample complexity bound for path graphs using a single logistic classifier. We follow a similar proof technique, but generalize this result to any graph and K binary logistic classifiers corresponding to vertices that form a locus cover. **We note that this theoretical setup is realistic, and it directly corresponds to our experiments using SimCLR and one-vs-rest logistic classifiers.** We provide an informal statement of the result and a sketch of the proof below. 

*Claim*

Let $G=(\mathcal{Y}, \mathcal{E})$ with $\mathcal{Y} = \{V_i\}_{i \in [m]}$ be an arbitrary graph with locus cover $\Lambda = \{V_i\}_{i \in [K]} \subseteq \mathcal{Y}$. 
Assume we have a one-vs-rest logistic regression classifier for each of the $K$ observed classes under the following realizability assumption for predicting an arbitrary class $V_*$:
$$
\mathbb{P}(y = V_i | x \text{ and } y \in \Lambda) =: \mathbb{P}_i 
= \frac{1}{1 + \exp\{-x^T\beta_i\}}
$$
where 
$$
V_* \in m_\Lambda([\mathbb{P_i}]_{i \in [K]}) = \argmin_{v \in \mathcal{Y}} \sum_{i \in [K]} \mathbb{P}_i d^2(v, v_i).
$$
We assume that we have access to the estimated logistic classifiers, $\widehat{\mathbb{P}}_i$ for all $i \in [K]$. 

Then the sample complexity of estimating $V_*$ using Loki is $O\left(d*\text{diam}(G)^2\right)$, where $d$ is the dimensionality of the input and $\text{diam}(G)$ is the graph diameter. 

*Proof sketch*

Our goal is to estimate $\mathbb{P}(y = V_* | x \text{ and } y \in \mathcal{Y})$ using the estimates $\widehat{\mathbb{P}}_i$. 
Loki yields 
$$
\widehat{\mathbb{P}}(y = V_* | x \text{ and } y \in \mathcal{Y}) = \mathbf{1}\{ \widehat{\mathbb{P}}_i \in [ \mathbb{P}_i - \epsilon_i, \mathbb{P}_i + \epsilon_i] \text{ for all } i \in [K] \}, 
$$
where $\epsilon_i \in O\left(\frac{1}{\text{diam}(G)}\right)$. 

We assume our estimates $\widehat{\mathbb{P}}_i$ over $n$ samples come from a joint distribution over the 2-norm unit ball and $\Lambda$, allowing us to use the standard finite sample estimation bound of each of the logistic regression parameters:
$||\widehat{\beta}_i - \beta_i|| \leq O\left(\sqrt{\frac{d}{n}}\right)$. From this, we obtain $||\widehat{\mathbb{P}}_i - \mathbb{P}|| \leq O\left(\sqrt{\frac{d}{n}}\right)$, which implies

$$
\widehat{\mathbb{P}}_i \in \left[\mathbb{P}_i - c\sqrt{\frac{d}{n}}, \mathbb{P}_i + c\sqrt{\frac{d}{n}}\right]. 
$$

Relating this interval to the prediction interval from before and adding the inequalities for all $K$, $\sum_{i \in [K]} \mathbb{P}_i - c\sqrt{\frac{d}{n}} \geq \sum_{i \in [K]} \mathbb{P}_i - \epsilon_i$, gives the desired result: 
$$
n \geq O(d * \text{diam}(G)^2). 
$$ 



**References**


[1] https://arxiv.org/abs/1503.08581 

[1.a] https://www.kaggle.com/competitions/lshtc 


[2] https://www.kaggle.com/datasets/owaiskhan9654/pubmed-multilabel-text-classification 


[3] https://arxiv.org/abs/2104.08821 
