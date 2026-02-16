# Automatic-Traceability
We propose an automated approach for visualizing the traceability between issue reports and the most relevant components for them, which combines Natural Language Processing, Deep Learning, and Graph Visualization techniques 

Installation
1. Open the project in Google Colab
2. Install the required libraries: Spacy, Transformers, Torch, Scikit-learn, Wordninja
3. Follow the step-by-step execution instructions
4. Test and review the results


Execution:

Run the projects in this order:

1. Classification/1_Classification_binary_2025.ipynb: Classifies the issue reports between containing_feature and non-containing_feature
2. Classificacion/2_Classification_bug_improv_new_feat_2025_.ipynb: Classifies the issue reports between Bug/Improvement/New Feature
3. Prediction/3_Prediction_components_2025.ipynb: Assign a software component to a issue report
4. Extraction/4_Extraction_text_feature_2025.ipynb: Extract main features of an issue report
5. Generating_graph/5_Generating_graph.ipynb: Generate and plot the Graph in format Pyvis

----------------------------------------------------------------------------------------------------------------------------------------------

Modern software development projects are based on incremental changes made to software components stored and tracked in public repositories such as GitHub. These change requests are stored as issue reports, which describe requests for implementing new features, fixing bugs or enhancing existing features.

Due to the increasing complexity of large-scale projects and the volume of issues reported daily during the software development and evolution, software engineers face the challenge to manually classify issue reports described using natural language into new feature, bug or improvement requests. Then, link them to the relevant component(s), and visualize the trace links between them. This manual effort is time-consuming and requires a deep knowledge of the project (source code and components), making it difficult to track software evolution 
effectively.

To overcome the challenge, several approaches use NLP techniques to address aspects related to the automatic classification of issue reports, but not sufficient to address aspects related to the traceability between a relevant component(s) and its reported issues.

In this work, we propose an automated approach for recovering and visualizing the traceability between issue reports and the most relevant components for them, which combines Natural Language Processing, BERT and Graph Visualization techniques for: 1) preprocessing issues by removing irrelevant information and tokenize them; 2) transform the tokens of issues into inputs for a BERT-based classifier then classifying issues into feature-containing and non-feature-containing, then sub-classifying non-feature containing issues into bugs and improvements using a BERT-based domain-specific language model; 3) extracting software features contained within issues using POS-tagging-based patterns and dependency parsing rules;  4) linking issues to the relevant predicted components (e.g. modules, packages or classes); and 5) generating a graph-based traceability visualization between a relevant component(s) and their reported issues. 

We show the feasibility of the proposed approach through the analysis of four projects described in the Dataset. Overall, our approach produced reasonable results, with accuracy greater than 93\%, outperforming the previous state-of-the-art approaches. 

<img src="https://github.com/francotejada/Automatic-Traceability/blob/main/Images/pipe_drawio.pdf" alt="Pipeline" >

<img src="https://github.com/francotejada/Automatic-Traceability/blob/main/Images/pipe_drawio.pdf" alt="Workflow" >

Our work allows researchers, as well as practitioners, to evolve components in a reasonable time and cost effective.


