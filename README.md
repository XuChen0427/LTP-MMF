The implementation of LTP-MMF in TOIS 2024, no other use of the code is allowed!

We here only provide steam dataset, other dataset please download them from the urls in the paepr

Note that due to the limitation space of anonymous.4open.science, we only simulate it using 256 users and 100 epochs, please modify the parameters for other settings

run the command:

python run_LTP-MMF.py

The result is: final NDCG:0.648 MMF:0.502 CTR:0.555

## For citation, please cite the following bib

```
@article{10.1145/3695867,
author = {Xu, Chen and Ye, Xiaopeng and Xu, Jun and Zhang, Xiao and Shen, Weiran and Wen, Ji-Rong},
title = {LTP-MMF: Towards Long-term Provider Max-min Fairness Under Recommendation Feedback Loops},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3695867},
doi = {10.1145/3695867},
abstract = {Multi-stakeholder recommender systems involve various roles, such as users, and providers. Previous work pointed out that max-min fairness (MMF) is a better metric to support weak providers. However, when considering MMF, the features or parameters of these roles vary over time, how to ensure long-term provider MMF has become a significant challenge. We observed that recommendation feedback loops (named RFL) will influence the provider MMF greatly in the long term. RFL means that recommender systems can only receive feedback on exposed items from users and update recommender models incrementally based on this feedback. When utilizing the feedback, the recommender model will regard the unexposed items as negative. In this way, the tail provider will not get the opportunity to be exposed, and its items will always be considered negative samples. Such phenomena will become more and more serious in RFL. To alleviate the problem, this paper proposes an online ranking model named Long-Term Provider Max-min Fairness (named LTP-MMF). Theoretical analysis shows that the long-term regret of LTP-MMF enjoys a sub-linear bound. Experimental results on three public recommendation benchmarks demonstrated that LTP-MMF can outperform the baselines in the long term.},
note = {Just Accepted},
journal = {ACM Trans. Inf. Syst.},
month = {sep},
keywords = {Max-min Fairness, Provider Fairness, Recommender System}
}
```
