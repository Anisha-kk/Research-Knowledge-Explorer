csv_content = """
title,year,topic
A Theory of the Learnable,1984,Traditional ML
Support-Vector Networks,1995,Traditional ML
Bagging Predictors,1996,Traditional ML
Random Forests,2001,Traditional ML
Latent Dirichlet Allocation,2003,Traditional ML
Greedy Function Approximation: A Gradient Boosting Machine,2001,Traditional ML
XGBoost: A Scalable Tree Boosting System,2016,Traditional ML
A Fast Learning Algorithm for Deep Belief Nets,2006,Deep Learning
Gradient-Based Learning Applied to Document Recognition,1998,Deep Learning
ImageNet Classification with Deep Convolutional Neural Networks,2012,Deep Learning
Dropout: A Simple Way to Prevent Neural Networks from Overfitting,2014,Deep Learning
Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,2015,Deep Learning
Deep Residual Learning for Image Recognition,2015,Deep Learning
Densely Connected Convolutional Networks,2017,Deep Learning
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,2019,Deep Learning
Neural Ordinary Differential Equations,2018,Deep Learning
Very Deep Convolutional Networks for Large-Scale Image Recognition,2014,Computer Vision
U-Net: Convolutional Networks for Biomedical Image Segmentation,2015,Computer Vision
You Only Look Once: Unified Real-Time Object Detection,2016,Computer Vision
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,2015,Computer Vision
Mask R-CNN,2017,Computer Vision
End-to-End Object Detection with Transformers,2020,Computer Vision
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,2020,Computer Vision
Segment Anything,2023,Computer Vision
DINOv2: Learning Robust Visual Features without Supervision,2023,Computer Vision
Playing Atari with Deep Reinforcement Learning,2013,Reinforcement Learning
Asynchronous Methods for Deep Reinforcement Learning,2016,Reinforcement Learning
Proximal Policy Optimization Algorithms,2017,Reinforcement Learning
Human-level Control Through Deep Reinforcement Learning,2015,Reinforcement Learning
Mastering the Game of Go with Deep Neural Networks and Tree Search,2016,Reinforcement Learning
Sequence to Sequence Learning with Neural Networks,2014,NLP
Neural Machine Translation by Jointly Learning to Align and Translate,2015,NLP
Attention Is All You Need,2017,Transformers
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,2018,LLM
Improving Language Understanding by Generative Pre-Training,2018,LLM
Language Models are Unsupervised Multitask Learners,2019,LLM
Language Models are Few-Shot Learners,2020,LLM
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,2020,LLM
Scaling Laws for Neural Language Models,2020,LLM
Training Compute-Optimal Large Language Models,2022,LLM
PaLM: Scaling Language Modeling with Pathways,2022,LLM
LLaMA: Open and Efficient Foundation Language Models,2023,LLM
Llama 2: Open Foundation and Fine-Tuned Chat Models,2023,LLM
Mistral 7B,2023,LLM
Gemini: A Family of Highly Capable Multimodal Models,2023,Foundation Models
Learning Transferable Visual Models From Natural Language Supervision,2021,Multimodal AI
Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision,2021,Multimodal AI
Flamingo: A Visual Language Model for Few-Shot Learning,2022,Multimodal AI
BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models,2023,Multimodal AI
Stable Diffusion: High-Resolution Image Synthesis with Latent Diffusion Models,2022,Generative AI
Hierarchical Text-Conditional Image Generation with CLIP Latents,2022,Generative AI
Diffusion Models Beat GANs on Image Synthesis,2021,Generative AI
Score-Based Generative Modeling through Stochastic Differential Equations,2021,Generative AI
Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context,2019,NLP
XLNet: Generalized Autoregressive Pretraining for Language Understanding,2019,NLP
RoBERTa: A Robustly Optimized BERT Pretraining Approach,2019,LLM
ALBERT: A Lite BERT for Self-supervised Learning of Language Representations,2019,LLM
ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators,2020,LLM
DeBERTa: Decoding-enhanced BERT with Disentangled Attention,2021,LLM
Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows,2021,Computer Vision
A ConvNet for the 2020s,2022,Computer Vision
Masked Autoencoders Are Scalable Vision Learners,2021,Computer Vision
A Simple Framework for Contrastive Learning of Visual Representations,2020,Self-Supervised Learning
Improved Baselines with Momentum Contrastive Learning,2020,Self-Supervised Learning
Bootstrap Your Own Latent,2020,Self-Supervised Learning
Emerging Properties in Self-Supervised Vision Transformers,2021,Self-Supervised Learning
Training Language Models to Follow Instructions with Human Feedback,2022,LLM Alignment
Constitutional AI: Harmlessness from AI Feedback,2022,LLM Alignment
Toolformer: Language Models Can Teach Themselves to Use Tools,2023,LLM Agents
ReAct: Synergizing Reasoning and Acting in Language Models,2022,LLM Agents
Chain-of-Thought Prompting Elicits Reasoning in Large Language Models,2022,LLM Reasoning
Tree of Thoughts: Deliberate Problem Solving with Large Language Models,2023,LLM Reasoning
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,2020,Applied AI
REALM: Retrieval-Augmented Language Model Pre-Training,2020,Applied AI
Improving Language Models by Retrieving From Trillions of Tokens,2021,Applied AI
A Generalist Agent,2022,Foundation Models
PaLM-E: An Embodied Multimodal Language Model,2023,Robotics + AI
RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,2023,Robotics + AI
Highly Accurate Protein Structure Prediction with AlphaFold,2021,Biological AI
Highly Accurate Protein Structure Prediction with AlphaFold-Multimer,2021,Biological AI
"""

with open("data/papers.csv", "w", encoding="utf-8") as f:
    f.write(csv_content)

print("papers.csv created successfully")