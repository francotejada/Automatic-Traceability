Repository: francotejada/automatic-traceability
File: 5_Generating_graph.ipynb
Lines: 305

Estimated tokens: 4.4k

Directory structure:
└── 5_Generating_graph.ipynb

"""
<a href="https://colab.research.google.com/github/francotejada/Automatic-Traceability/blob/main/Generating_graph/5_Generating_graph.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

!pip install pyvis

from pyvis.network import Network
from IPython.display import HTML

# Crear red
net = Network(
    notebook=True,
    directed=True,
    height="500px",
    width="100%",
    cdn_resources="in_line"   # 👈 SOLUCIÓN
)


import re
import os
from pyvis.network import Network
from IPython.display import HTML

# 1. Configuración del objeto net como solicitaste
net = Network(notebook=True, cdn_resources='in_line', height="750px", width="100%", bgcolor='#222222', font_color='white')

def procesar_archivo_a_pyvis(file_path):
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    # --- LIMPIEZA DINÁMICA ---
    # Eliminar etiquetas tipo
    content = re.sub(r'\\', '', raw_content)
    # Unir líneas que se rompen por los comentarios de origen y limpiar barras
    content = content.replace('\n', ' ').replace('\\', '')
    # Normalizar espacios múltiples
    content = re.sub(r'\s+', ' ', content)

    # --- EXTRACCIÓN CON REGEX ---
    # Patrón para Nodos: create (variable:tipo {name:'...', color...})
    node_pattern = r"create\s*\(([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)\s*\{name:\s*'([^']*)'(?:,\s*favoritecolor:\s*'([^']*)')?\s*\}\s*\)"

    # Patrón para Relaciones: create (origen)-[:TIPO]->(destino)
    edge_pattern = r"create\s*\(([a-zA-Z0-9_]+)\)\s*-\[:([a-zA-Z0-9_]+)\]->\s*\(([a-zA-Z0-9_]+)\)"

    # --- CARGA DE NODOS ---
    # Usamos un set para evitar errores si el script intenta crear el mismo nodo dos veces
    nodos_creados = set()
    for match in re.finditer(node_pattern, content):
        var_id, n_type, name, fav_color = match.groups()

        if var_id not in nodos_creados:
            # Lógica de colores: Clases en verde, Issues en naranja
            color = 'green' if (fav_color == 'green' or n_type == 'class') else '#FFA500'
            shape = 'ellipse' if n_type == 'class' else 'dot'

            net.add_node(var_id, label=name, color=color, shape=shape, title=f"Type: {n_type}")
            nodos_creados.add(var_id)

    # --- CARGA DE RELACIONES ---
    for match in re.finditer(edge_pattern, content):
        source, rel_type, target = match.groups()
        # Verificamos que los nodos existan antes de conectarlos
        if source in nodos_creados and target in nodos_creados:
            net.add_edge(source, target, label=rel_type, color='#848484')

    # --- RENDERIZADO ---
    # En Colab, pyvis requiere un nombre de archivo para el método show()
    output_name = "grafo_jbehave.html"
    net.show(output_name)
    return HTML(output_name)

# 2. Ejecutar el proceso
# Cambia 'jbehave node4j part.txt' por el nombre real de tu archivo en Colab
procesar_archivo_a_pyvis('jbehave_node4j_part.txt')
# Output:
#   grafo_jbehave.html

#   <IPython.core.display.HTML object>

HTML(net.generate_html())
# Output:
#   <IPython.core.display.HTML object>

from pyvis.network import Network

# Inicializar la red
net = Network(notebook=True, cdn_resources='in_line', height="750px", width="100%")

# --- NODOS ---
# Nodos tipo 'class' (color verde)
net.add_node("hudson", label="hudson", color="green", title="Class")
net.add_node("configuration", label="configuration", color="green", title="Class")
net.add_node("scenario", label="scenario", color="green", title="Class")
net.add_node("GivenStories", label="GivenStories", color="green", title="Class")
net.add_node("ant", label="ant ", color="green", title="Class")
net.add_node("maven", label="maven", color="green", title="Class")
net.add_node("StoryReporterBuilder", label="StoryReporterBuilder", color="green", title="Class")
net.add_node("print", label="print", color="green", title="Class")
net.add_node("pending", label="pending", color="green", title="Class")
net.add_node("SpringStoryReporterBuilder", label="SpringStoryReporterBuilder", color="green", title="Class")
net.add_node("core", label="core", color="green", title="Class")
net.add_node("initialize", label="initialize", color="green", title="Class")

# Nodos tipo 'issue' (color por defecto o azul)
net.add_node("Hudson_xUnit", label="Hudson xUnit", title="Issue")
net.add_node("Thanks_for_help", label="Thanks for help", title="Issue")
net.add_node("convert_file", label="convert file", title="Issue")
net.add_node("know_about_similar_tasks", label="know about similar tasks", title="Issue")
net.add_node("takes_place_with_version", label="takes place with version", title="Issue")
net.add_node("change_users", label="change users", title="Issue")
net.add_node("has_installed_ant_version_in_case", label="has installed ant version in case", title="Issue")
net.add_node("Stack_Traces", label="Stack Traces", title="Issue")
net.add_node("has_three_lines", label="has three lines", title="Issue")
net.add_node("be_in_stack", label="be in stack", title="Issue")
net.add_node("has_three_different_fragments_of_stack", label="has three different fragments of stack", title="Issue")
net.add_node("Stack_Traces_long", label="Stack Traces - printed output...", title="Issue") # Nombre abreviado para legibilidad en código
net.add_node("constituent_scenarios", label="constituent scenarios", title="Issue")
net.add_node("stories_with_steps", label="stories with steps", title="Issue")
net.add_node("view_scenarios", label="view scenarios", title="Issue")
net.add_node("mean_in_kind", label="mean in kind", title="Issue")
net.add_node("maintain_flow_of_top_development", label="maintain flow of top development", title="Issue")
net.add_node("expose_properties", label="expose properties", title="Issue")
net.add_node("expose_properties_in_StoryReporterBuilder", label="expose properties in StoryReporterBuilder", title="Issue")
net.add_node("expose_all_properties_long", label="expose all properties...", title="Issue")
net.add_node("Initialisation_errors", label="Initialisation errors", title="Issue")
net.add_node("Initialisation_errors_in_RemoteWebDriverProvider", label="Initialisation errors in RemoteWebDriverProvider", title="Issue")
net.add_node("verify_behaviour", label="verify behaviour", title="Issue")
net.add_node("lead_to_RunningStoriesFailed_exception", label="lead to RunningStoriesFailed exception", title="Issue")
net.add_node("Initialisation_errors_long", label="Initialisation errors - errors in RemoteWebDriverProvider...", title="Issue")
net.add_node("click_story_name", label="click story name", title="Issue")
net.add_node("Story_source_HTMLified_long", label="Story source - HTMLified story...", title="Issue")
net.add_node("allow_output", label="allow output", title="Issue")
net.add_node("write_to_one_format", label="write to one format", title="Issue")
net.add_node("allow_output_of_JSON_data", label="allow output of JSON data", title="Issue")
net.add_node("allow_output_long", label="allow output - allow output of JSON...", title="Issue")
net.add_node("NPE_NPE", label="NPE NPE", title="Issue")
net.add_node("Error_in_WebDriverProvider_initialize_method", label="Error in WebDriverProvider initialize method", title="Issue")
net.add_node("throw_new_RuntimeException", label="throw new RuntimeException", title="Issue")
net.add_node("is_in_StoryRunner", label="is in StoryRunner", title="Issue")
net.add_node("Error_in_WebDriverProvider_long", label="Error in WebDriverProvider...", title="Issue")


# --- ARISTAS (RELACIONES) ---

# Grupo Hudson
net.add_edge("Hudson_xUnit", "hudson", label="Bug")
net.add_edge("Thanks_for_help", "hudson", label="Bug")
net.add_edge("convert_file", "hudson", label="Improvement")
net.add_edge("know_about_similar_tasks", "hudson", label="Improvement")
net.add_edge("takes_place_with_version", "hudson", label="Improvement")

# Grupo Configuration
net.add_edge("Hudson_xUnit", "configuration", label="Bug")
net.add_edge("Thanks_for_help", "configuration", label="Bug")
net.add_edge("convert_file", "configuration", label="Improvement")
net.add_edge("know_about_similar_tasks", "configuration", label="Improvement")
net.add_edge("takes_place_with_version", "configuration", label="Improvement")
net.add_edge("allow_output", "configuration", label="Bug")
net.add_edge("write_to_one_format", "configuration", label="Bug")
net.add_edge("allow_output_of_JSON_data", "configuration", label="Improvement")
net.add_edge("allow_output_long", "configuration", label="Improvement")

# Grupo Scenario
net.add_edge("Hudson_xUnit", "scenario", label="Bug")
net.add_edge("Thanks_for_help", "scenario", label="Bug")
net.add_edge("convert_file", "scenario", label="Improvement")
net.add_edge("know_about_similar_tasks", "scenario", label="Improvement")
net.add_edge("takes_place_with_version", "scenario", label="Improvement")
net.add_edge("constituent_scenarios", "scenario", label="Bug")
net.add_edge("stories_with_steps", "scenario", label="Bug")
net.add_edge("view_scenarios", "scenario", label="Improvement")
net.add_edge("mean_in_kind", "scenario", label="Improvement")
net.add_edge("maintain_flow_of_top_development", "scenario", label="Improvement")

# Grupo GivenStories
net.add_edge("Hudson_xUnit", "GivenStories", label="Bug")
net.add_edge("Thanks_for_help", "GivenStories", label="Bug")
net.add_edge("convert_file", "GivenStories", label="Improvement")
net.add_edge("know_about_similar_tasks", "GivenStories", label="Improvement")
net.add_edge("takes_place_with_version", "GivenStories", label="Improvement")

# Grupo Ant / Maven
net.add_edge("change_users", "ant", label="Improvement")
net.add_edge("has_installed_ant_version_in_case", "ant", label="Improvement")
net.add_edge("click_story_name", "ant", label="Bug")
net.add_edge("Story_source_HTMLified_long", "ant", label="Bug")
net.add_edge("change_users", "maven", label="Improvement")
net.add_edge("has_installed_ant_version_in_case", "maven", label="Improvement")

# Grupo StoryReporterBuilder
net.add_edge("Stack_Traces", "StoryReporterBuilder", label="Bug")
net.add_edge("has_three_lines", "StoryReporterBuilder", label="Bug")
net.add_edge("be_in_stack", "StoryReporterBuilder", label="Improvement")
net.add_edge("has_three_different_fragments_of_stack", "StoryReporterBuilder", label="Improvement")
net.add_edge("Stack_Traces_long", "StoryReporterBuilder", label="Improvement")
net.add_edge("expose_properties", "StoryReporterBuilder", label="Bug")
net.add_edge("expose_properties_in_StoryReporterBuilder", "StoryReporterBuilder", label="Improvement")
net.add_edge("expose_all_properties_long", "StoryReporterBuilder", label="Improvement")

# Grupo Print
net.add_edge("Stack_Traces", "print", label="Bug")
net.add_edge("has_three_lines", "print", label="Bug")
net.add_edge("be_in_stack", "print", label="Improvement")
net.add_edge("has_three_different_fragments_of_stack", "print", label="Improvement")
net.add_edge("Stack_Traces_long", "print", label="Improvement")

# Grupo Pending
net.add_edge("constituent_scenarios", "pending", label="Bug")
net.add_edge("stories_with_steps", "pending", label="Bug")
net.add_edge("view_scenarios", "pending", label="Improvement")
net.add_edge("mean_in_kind", "pending", label="Improvement")
net.add_edge("maintain_flow_of_top_development", "pending", label="Improvement")

# Grupo SpringStoryReporterBuilder / Core
net.add_edge("expose_properties", "SpringStoryReporterBuilder", label="Bug")
net.add_edge("expose_properties_in_StoryReporterBuilder", "SpringStoryReporterBuilder", label="Improvement")
net.add_edge("expose_all_properties_long", "SpringStoryReporterBuilder", label="Improvement")
net.add_edge("expose_properties", "core", label="Bug")
net.add_edge("expose_properties_in_StoryReporterBuilder", "core", label="Improvement")
net.add_edge("expose_all_properties_long", "core", label="Improvement")

# Grupo Initialize
net.add_edge("Initialisation_errors", "initialize", label="Bug")
net.add_edge("Initialisation_errors_in_RemoteWebDriverProvider", "initialize", label="Bug")
net.add_edge("verify_behaviour", "initialize", label="Improvement")
net.add_edge("lead_to_RunningStoriesFailed_exception", "initialize", label="Improvement")
net.add_edge("Initialisation_errors_long", "initialize", label="Improvement")
net.add_edge("NPE_NPE", "initialize", label="Bug")
net.add_edge("Error_in_WebDriverProvider_initialize_method", "initialize", label="Bug")
net.add_edge("throw_new_RuntimeException", "initialize", label="Improvement")
net.add_edge("is_in_StoryRunner", "initialize", label="Improvement")
net.add_edge("Error_in_WebDriverProvider_long", "initialize", label="Improvement")

# Mostrar opciones de física para interactuar
net.show_buttons(filter_=['physics'])

# Generar archivo HTML
#net.show("grafo_issues.html")

HTML(net.generate_html())
# Output:
#   <IPython.core.display.HTML object>

