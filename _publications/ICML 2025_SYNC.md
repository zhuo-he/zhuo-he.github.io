---
title: "Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains"
collection: publications
category: conferences
permalink: /publication/Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains
excerpt: 'This paper propose SYNC, a novel method for evolving domain generalization that learns time-aware causal representations by modeling dynamic causal factors and mechanism drifts, achieving robust generalization across temporal domains.'
date: 2025-05-01
venue: 'International Conference on Machine Learning'
paperurl: 'https://arxiv.org/abs/2506.17718'
codeurl: 'https://github.com/BIT-DA/SYNC'
---

![SYNC](https://github.com/zhuo-he/zhuo-he.github.io/blob/master/images/sync_framework.png)

Abstract: Endowing deep models with the ability to generalize in dynamic scenarios is of vital significance for real-world deployment, given the continuous and complex changes in data distribution. Recently, evolving domain generalization (EDG) has emerged to address distribution shifts over time, aiming to capture evolving patterns for improved model generalization. However, existing EDG methods may suffer from spurious correlations by modeling only the dependence between data and targets across domains, creating a shortcut between task-irrelevant factors and the target, which hinders generalization. To this end, we design a time-aware structural causal model (SCM) that incorporates dynamic causal factors and the causal mechanism drifts, and propose **S**tatic-D**YN**amic **C**ausal Representation Learning (**SYNC**), an approach that effectively learns time-aware causal representations. Specifically, it integrates specially designed information-theoretic objectives into a sequential VAE framework which captures evolving patterns, and produces the desired representations by preserving intra-class compactness of causal factors both across and within domains. Moreover, we theoretically show that our method can yield the optimal causal predictor for each time domain. Results on both synthetic and real-world datasets exhibit that SYNC can achieve superior temporal generalization performance.

[Paper](https://arxiv.org/abs/2506.17718) [Code](https://github.com/BIT-DA/SYNC)

