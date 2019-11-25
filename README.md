# Active-Learning-on-Regression
 Regression problems are pervasive in real-world applications. Generally a substantial amount of labeled samples are needed to build a regression model with good general- ization ability. However, many times it is relatively easy to collect a large number of un- labeled samples, but time-consuming or expensive to label them. Active learning for re- gression (ALR) is a methodology to reduce the number of labeled samples, by selecting the most beneficial ones to label, instead of random selection. This paper proposes two new ALR approaches based on greedy sampling (GS). The first approach (GSy) selects new samples to increase the diversity in the output space, and the second (iGS) selects new samples to increase the diversity in both input and output spaces. Extensive experiments on 10 UCI and CMU StatLib datasets from various domains, and on 15 subjects on EEG- based driver drowsiness estimation, verified their effectiveness and robustness.