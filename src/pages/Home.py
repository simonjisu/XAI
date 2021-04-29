import streamlit as st
from pathlib import Path
from PIL import Image
from .DownDataset import download_dataset

def write():
    repo_path = Path(".").absolute()
    assets_path = repo_path / "assets"
    st.title("eXplainable AI in Vision")
    st.markdown(
    """
    **[Warning]** This article is collection of some good articles that i read, 
    so it contains a lot of references. Please visit them to learn more about XAI.

    It may take some times when you first run this app, since it needs to download the MNIST and CIFAR10 dataset.

    Please enjoy the demo. 

    :envelope: Contact: simonjisu@gmail.com
    """
    )
    img = Image.open(assets_path / "monocle_pixabay.png")
    st.image(img, caption="Source: [Pixabay](https://pixabay.com/ko/vectors/%EA%B0%9C-%ED%86%B1-%ED%96%87-%EC%95%A0%EC%99%84-%EB%8F%99%EB%AC%BC-k9-4311330/)")
    eval_type_options = {
        "mnist": ["roar", "kar"],
        "cifar10": ["roar-plain", "kar-rcd", "roar-rcd", "roar-rcd-fgm", "roar-rcd-noabs"]
    }
    download_dataset(eval_type_options)

    st.info(":point_left: Click the sidebar to navigate the some experiment demo.")
    sidebar_container = st.sidebar.beta_container()
    with sidebar_container:
        st.markdown(
        """
        ** Contents **

        1. [XAI](#1-xai)
        2. [Interpretability](#2-interpretability)
            - [What is Interpretability](#what-is-Interpretability)
            - [Why Interpretability](why-interpretability)
            - [Types of Interpretability in ML](#types-of-interpretability-in-ml)
        3. [Deep Neural Network Interpretability](#3-deep-neural-network-interpretability)
            - [Interpreting Models](#interpreting-models-macroscopic)
                - Representation Analysis
                - Data Generation
                - Example-based
            - [Interpreting Descions](#interpreting-descions-microscopic)
                - Example-based
                - Attribution Methods
        4. [Evaluation of Attribution Methods](#4-evaluation-of-attribution-methods)
            - [Qualitative](#qualitative)
                - Coherence
                - Class Sensitivity
            - [Quantitative](#quantitative)
                - Selectivity
                - ROAR / KAR
        """)
    
    st.header("References")
    st.markdown(
    """
    * 1-1: [Wikipedia: Explainable_artificial_intelligence](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
    * 2-1: [Explanation in Artificial Intelligence: Insights from the Social Sciences](https://arxiv.org/abs/1706.07269)
    * 2-2: [Explaining Explanations: An Overview of Interpretability of Machine Learning](https://arxiv.org/abs/1806.00069)
    * 3-1: [A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758)
    """)
    st.subheader("Attribution Methods Papers:")
    st.markdown(
    """
    * [Not Just a Black Box: Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1605.01713)
    * [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
    * [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """)
    st.header("Useful Articles")
    st.markdown(
    """
    * [Molnar, Christoph. "Interpretable machine learning. A Guide for Making Black Box Models Explainable", 2019](https://christophm.github.io/interpretable-ml-book/)
    * [Explainable AI (Part I): Explanations and Opportunities](https://dainstudios.com/2019/10/24/explainable-ai-part-1/)
    * [Korean Review for References 1-2](https://simonjisu.github.io/paper/2019/09/18/introxai.html)
    
    ---
    """)

    st.header("1. XAI")
    st.markdown(
    """
    Explainable AI (XAI) refers to methods and techniques in the application of artificial intelligence technology (AI) such that the results of the solution can be understood by human experts. It contrasts with the concept of the "black box" in machine learning where even their designers cannot explain why the AI arrived at a specific decision.
    
    * From [References: 1-1](#references)
    """)

    st.header("2. Interpretability")
    st.subheader("What is Interpretability?")
    st.markdown(
    """
    I use [Biran and Cotton’s](https://pdfs.semanticscholar.org/02e2/e79a77d8aabc1af1900ac80ceebac20abde4.pdf) definition of **interpretability** of a model as: the degree to which an observer can understand the cause of a decision.
    
    * From: [References: 2-1](#references)

    > **The Goal of Interpretability**
    
    The goal of interpretability is to describe the internals of a system in a way that is understandable to humans. The success of this goal is tied to the cognition, knowledge, and biases of the user: for a system to be interpretable, it must produce descriptions that are simple enough for a person to understand using a vocabulary that is meaningful to the user.
    
    > **What is an Explanation?**
    
    Philosophical texts show much debate over what constitutes an explanation. Of particular interest is what makes an explanation 
    "good enough" or what really defines an explanation. Some say a good explanation depends on the question. This set of essays 
    discusses the nature of explanation, theory, and the foundations of linguistics. Although for our work, the most important and 
    interesting work is on "Why questions." In particular, when you can phrase what you want to know from an algorithm as a why question, 
    there is a natural qualitative representation of when you have answered said question–when you can no longer keep asking why. 
    There are two why questions of interest; why and why-should. Similarly to the explainable planning literature, philosophers 
    wonder about the why-shouldn’t and why-should questions, which can give the kinds of explainability requirements we want. 
    
    * From: [References: 2-2](#references)
    """)
    st.subheader("Why Interpretability?")
    st.markdown(
    """
    1. Verify that model works as expected
        * Wrong decisions can be costly and dangerous: [Self-driving Uber kills Arizona woman in first fatal crash involving pedestrian](https://www.theguardian.com/technology/2018/mar/19/uber-self-driving-car-kills-woman-arizona-tempe) / Disease Misclassification
    2. Improve / Debug classifier
    3. Make new discoveries: 
        * Learn about the physical / biological / chemical mechanisms
        * Learn about the human brain
    4. Right to explanation
        * US Equal Credit Opportunity Act
        * The European Union General Data Protection Regulation
        * France Digital Republic Act
    
    * From: Taegyun Jeon(SIA, Founder and Chief Executive Office) in the lecture of AICollege (Not for Share)
    """)
    st.subheader("Types of Interpretability in ML")
    st.markdown(
    """
    * Ante-hoc Interpretability: Choose an interpretable model and train it.
        * Example: Decision Tree
        * Problem: Is the model expressive enough to predict the data?
    * Post-hoc Interpretability: Choose a complex model and develop a special technique to interpret it.
        * Deep Neural Networks
        * How to interpret millions of parameters?
    
    * From: The lecture of Taegyun Jeon(SIA, Founder and Chief Executive Office) in AICollege (Not for Share)

    ---
    """)
    st.header("3. Deep Neural Network Interpretability")
    st.subheader("Interpreting Models(Macroscopic)")
    st.markdown(
    """
    Better understand internal representations
    * "Summarize" DNN with a simpler model (e.g. decision tree) 
    * Find prototypical example of a category
    * Find pattern maximizing activation of a neuron
    
    Types of Interpretability
    * Representation Analysis
    * Data Generation
    * Example-based
    
    * From: The lecture of Taegyun Jeon(SIA, Founder and Chief Executive Office) in AICollege (Not for Share)
    """)
    st.subheader("Interpreting Descions(Microscopic)")
    st.markdown(
    """
    Important for practical applications
    - Why did DNN make this decision?
    - Verify that model behaves as expected - Find evidence for decision
    
    Types of Interpretability
    * Example-based
    * Attribution Methods
    
    * From: The lecture of Taegyun Jeon(SIA, Founder and Chief Executive Office) in AICollege (Not for Share)

    ---
    """)
    st.header("4. Evaluation of Attribution Methods")
    st.subheader("Qualitative")
    st.markdown(
    """
    - Coherence
    - Class Sensitivity
    """)
    st.subheader("Quantitative")
    st.markdown(
    """
    > **Selectivity**

    Process:

    1. Evaluation with attribution method and get the attribution scores for the input datas. 
    2. Ranking the attribution scores and replace the input datas to 0 according to the ranked attribution scores with the percentage. 
        Call them as "perturbed datas"
    3. Ust the "perturbed datas" as a new input datas for model.
    4. Repeat 1-3 process for a certain steps that you set. Also, record the metrics at each step.
    
    > **ROAR / KAR**

    Process:

    1. Evaluation with attribution method and get the attribution scores for the input datas.
    2. Ranking the attribution scores, after this using a small percentage(from 10% ~ 90%)
        * [ROAR] replace the input datas to 0 according to the ranked attribution scores with the percentage.
        * [KAR] begins with a datas filled with zeros, fill original datas according to the ranked attribution scores with the percentage.
        * Call them as "perturbed datas"
    3. Ust the "perturbed datas" as a new input datas for model. 
    4. Retrain the model and record the metrics.

    **What would happen without re-training?**
    >
    > The re-training is the most computationally expensive aspect of ROAR. One should question whether it is actually needed. 
    We argue that re-training is needed because machine learning models **typically assume that the train and the test data comes 
    from a similar distribution.** The replacement value c can only be considered uninformative if the model is trained to learn 
    it as such. Without retraining, it is unclear whether degradation in performance is **due to the introduction of artifacts 
    outside of the original training distribution** or **because we actually removed information**. This is made explicit in our 
    experiment in Section 4.3.1, we show that without retraining the degradation is far higher than the modest decrease in performance 
    observed with re-training. This suggests retraining has better controlled for artefacts introduced by the modification.
    >
    > * From: [References: 3-1](#references)
        
    ---
    """)
    