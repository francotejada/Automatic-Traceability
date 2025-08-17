# Automatic-Traceability
We propose an automated approach for visualizing the traceability between issue reports and the most relevant components for them, which combines Natural Language Processing, Deep Learning, and Graph Visualization techniques 

Installation
1. Open the project in Google Colab
2. Install the required libraries: Spacy, Transformers, Torch, Scikit-learn, Wordninja
3. Follow the step-by-step execution instructions
4. Test and review the results


Execution:

Run the projects in this order:

1. Clasificacion/1_Clasificacion_binaria_2025.ipynb: Classifies the issue reports between containing_feature and non-containing_feature
2. Clasificacion/2_Clasificacion_bug_improv_new_feat_2025_.ipynb: Classifies the issue reports between Bug/Improvement/New Feature
3. Prediccion/3_Prediccion_componentes_2025.ipynb: Assign a software component to a issue report
4. Extraccion/4_Extraccion_caracteristicas_texto_2025.ipynb: Extract main features of an issue report
