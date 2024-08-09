# Research Lists in LLM

## Widely used open-sourced LLM  


- **LLM with the longest context: XVERSE-13B-256K**  
**Website:** https://huggingface.co/xverse/XVERSE-13B-256K  

- **The strongest bilingual language model: 悟道天鹰-34B**  
**Website:** https://zhuanlan.zhihu.com/p/661190270  

- **Google Gemini (mainly multimodal model? Perform poorly in AM)**  
**paper:** Gemini:A Family of Highly Capable Multimodal Models (Google deepmind)    

- **Falcon**
**Website:** https://huggingface.co/blog/falcon

- **LLama 1/2/3**  
**paper:** LLaMA: Open and Efficient Foundation Language Models (Meta AI)  

- **智源研究院: Aquila2-34B/7B AquilaChat2-34B/7B AquilaSQL**  
An efficient training framework mentioned, support training in parallel.   


## works of fine-tuning model  

- **Evaluating Large Language Models Trained on Code (function synthesis on coding tasks)**  
**Github:** https://github.com/openai/human-eval  
**Paper:** https://arxiv.org/pdf/2107.03374  

- **LLama Chinese:**  
**Websites:** https://zhuanlan.zhihu.com/p/656696049?utm_id=0  https://blog.csdn.net/yuyangchenhao/article/details/131290469  
**Training tricks:** https://zhuanlan.zhihu.com/p/631360711  https://zhuanlan.zhihu.com/p/635931929?utm_id=0   
**Github:** https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main  

- **ORLM: Training Large Language Models for Optimization Modeling (on LLama3-8B)**  

- **Stanford Alpaca: An Instruction-following LLaMA Model**  
**Github:** https://github.com/tatsu-lab/stanford_alpaca  

- **LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation**


## works of pretrain  

- **LLama 65B**  
**Websotes:** https://baijiahao.baidu.com/s?id=1771756789663508724&wfr=spider&for=pc   

## prompt engineering  

- **PAL: Program-aided Language Models**  

- **LARGE LANGUAGE MODELS AS OPTIMIZERS (Google Deepmind)**  
meta prompt  

- **Solving Math Word Problems by Combining Language Models With Symbolic Solvers (Stanford University)**  
Declarative prompt  

- **OPTIMUS: OPTIMIZATION MODELING USING MIP SOLVERS AND LARGE LANGUAGE MODELS (Stanford University)**  
Including code generation  

- **Diagnosing Infeasible Optimization Problems Using Large Language Models (Purdue University)**    
using Gurobi   

- **Solving Math Word Problems by Combining Language Models With Symbolic Solvers**  

- **LLM-Resistant Math Word Problem Generation via Adversarial Attacks**  

- **DESIGN OF CHAIN-OF-THOUGHT IN MATH PROBLEMSOLVING**  

- **FILL IN THE BLANK: EXPLORING AND ENHANCING LLM CAPABILITIES FOR BACKWARD REASONING INMATH WORD PROBLEMS**  

- **OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models**  

- **CHAIN-OF-EXPERTS: WHEN LLMS MEET COMPLEXOPERATIONS RESEARCH PROBLEMS**  

## RAG  

- **From Local to Global: A Graph RAG Approach to query-focused summarization (Microsoft)**  

- **FoRAG: Factuality-optimized Retrieval Augmented Generation for Web-enhanced Long-form Question Answering**  

- **FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research**  





## Tools  

- **Token estimization:** https://toolai.fun/  

- **Langchain:** https://www.163.com/dy/article/IJ4O5FRG0531FIPF.html  

## Some heuristic issues  

- **Difference between GPT & Claude:** https://blog.csdn.net/u011537073/article/details/133950059  

- **Construct LLM:** https://zhuanlan.zhihu.com/p/664046612?utm_psn=1702755988845731840    


## LLM summary related to optimization problem (with detailed information)

### - **ORLM: Training Large Language Models for Optimization Modeling**
Motivation: heavily rely on prompt engineering (regard multi-agent cooperation as prompt similarly), causing privacy concerns. Based on this, author proposes to train the open-source LLMs for optimization modeling.
#### Introduction
The top-3 lines indicate some prior works on automating optimization modeling.
Page 2: Previous research has focused on ... reference paper
pretrained-LMs result in accumulated errors and have limited generalization capabilities (because of relatively-small parameter scale)
OptiGuide: Large Language Models for Supply Chain Optimization
Former algorithms rely heavily on sensitive data? Is that true?

Strongly data requirements

There exist multiple modeling techniques for the same problem, including these in the dataset would enable the model to learn various modeling skills. How to train LLM with different output?

Using seed data to generate more data.

#### Background and Desiderata
Employ COPT [Cardinal Optimizer (COPT) user guide. https://guide.coap.online/copt/en-doc] as the default program solver.


Different scenarios, Different problem types, Different difficuilty levels.
One would expect diversity at the problem-solution level? How to achieve this?
The constructed dataset itself should contain linguistic diversity.

Multiple modeling techniques, such as linearizing nonlinear problem by introducing auxiliary variables. Including this variety in the dataset allows the model to learn diffenrent modeling techniques and approaches.

A referenct paper in 3.1: A synthesis method: Self-instruct: Aligning language models with self-generated instructions. It is aimed at expanding tasks.

Expanding scenarios partially addresses comprehensive coverage issue in terms of scenerios and question types? (Why claiming this?) It falls short in varying levels of difficulty (manually reviewing the difficulty).

Section 3.2: Instructing GPT-4 to rewrite the OR problem, either simplifying or complicating it. The aim of this process is to ensure the core logic aligns with the solution.

section 3.3: Filtering: 1. Remove examples where questions duplicate each other or match questions in the evaluation benchmarks.
                        2. Manually correct minor grammatical errors (maybe caused by the unfamiliarity of GPT-4 with the COPT API).

Experiments:

Using stanford-alpaca-like template as the input prompt.
Only compute the loss over the target completion.
Greedy decoding under 0-shot setting to eliminate randomness, selecting the top-1 completion as the final solution.
Converting the mathematical problems into executable programs using GPT-4, and use these as ground truth.

Dataset: 1. NL4OPT: NL4Opt Competition: Formulating Optimization Problems Based on Their Natural Language Descriptions
         2. MAMO: Mamo: a mathematical modeling benchmark with solvers
         3. IndustryOR: This paper.

The performance can be measured using execution accuracy, where an executed optimal value that matches any provided GT optimal value is considered correct.

A reference: Reflexion: Reflexion: an autonomous agent with dynamic memory and self-reflection

#### Code remains to be tested


### - **OceanGPT: A large language model for ocean science tasks**
Motivation: The potential of LLMs for ocean science is under-explored.
The difficulty is mainly on the immense and intricate nature of ocean data as well as the richness in knowledge.

Contribution abstract: Propose DoInstruct, a novel framework to automatically obtain a large volume of ocean domain instruction data, which generates instructions based on **multi-agent collaboration**.
To sum up, the main hurdle is the data and the specific knowledge.

Introduction: Agents are assigned with different role in a specific domain (science and research, resources and development, ecology and environment) and responsible forgenerating the corresponding data.

**Related work, Science large language models,** a number of works in various domains are listed.

#### Method:

traning corpus + pretrain + finetune (use llama-2)

##### Pretrain
- Collect 67633 documents from **open-access literature** (hahahah authors emphasis this in paper).
- For diversity, authors choose articles from different sources to ensure coverage of various research perspectives and methods.
- For pdf transformation, employ pdfminer to convert the content of literature files into plain text.
- Apply regular expressions to filter out figures, tables, headers, footers, page numbers, URLs and references. Additionally remove extra spaces, line breaks and non-text characters.
- Employ **hash-based methods** to de-duplicate the data.


















