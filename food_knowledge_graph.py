# ============================================
# LIGHTWEIGHT FOOD KNOWLEDGE GRAPH + SPARQL (rdflib)
# ============================================
# Inspired by FoCOSA (Food Consumer-Oriented Sustainability-Aware ontology)
# and KG-based diet recommendation papers (KG-DietNet, DietQA, Graph-RAG),
# but scoped honestly to what our FCTB data actually supports -- a compact
# custom schema (not FoCOSA's full 41-concept model), built with rdflib so
# no external graph database server (e.g. Neo4j) is required.
#
# Classes  : Food, Role, MealType, GICategory, DietCluster
# Properties: hasRole, hasMealType, hasGICategory, hasDietCluster,
#             hasEnergy, hasCarbohydrate, isStandalone, hasBengaliName

import pandas as pd
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD
from rdflib.namespace import NamespaceManager
import re

print("=== STEP 1: Loading FCTB dataset ===")
df = pd.read_csv("diabetes_prediction/diet_data/fctb_clustered.csv")
print(f"Loaded {df.shape[0]} food items")

# ── 2. DEFINE NAMESPACE / SCHEMA ────────────────
FOOD = Namespace("http://example.org/food-ontology#")
g = Graph()
g.bind("food", FOOD)

# Declare the schema (classes + properties) explicitly, like a mini-ontology
for cls in ["Food", "Role", "MealType", "GICategory", "DietCluster"]:
    g.add((FOOD[cls], RDF.type, RDFS.Class))
for prop in ["hasRole", "hasMealType", "hasGICategory", "hasDietCluster",
             "hasEnergy", "hasCarbohydrate", "isStandalone", "hasBengaliName"]:
    g.add((FOOD[prop], RDF.type, RDF.Property))

def slugify(name):
    """Turn a food name into a safe URI fragment."""
    name = re.sub(r"[^\w\s-]", "", str(name))
    name = re.sub(r"\s+", "_", name.strip())
    return name[:60]

# ── 3. POPULATE THE GRAPH FROM FCTB ─────────────
print("\n=== STEP 2: Converting FCTB rows into RDF triples ===")
for _, row in df.iterrows():
    food_uri = FOOD[f"food_{row['Code']}_{slugify(row['Food_Name_English'])}"]
    g.add((food_uri, RDF.type, FOOD.Food))
    g.add((food_uri, RDFS.label, Literal(row["Food_Name_English"])))
    g.add((food_uri, FOOD.hasBengaliName, Literal(row["Food_Name_Bengali"])))
    g.add((food_uri, FOOD.hasRole, FOOD[str(row["Food_Role"]).replace(" ", "_")]))
    g.add((food_uri, FOOD.hasMealType, FOOD[str(row["Meal_Type"]).replace("/", "_")]))
    g.add((food_uri, FOOD.hasGICategory, FOOD[str(row["GI_Category"])]))
    g.add((food_uri, FOOD.hasDietCluster, FOOD[str(row["Diet_Cluster"]).replace(" ", "_").replace("-", "_")]))
    g.add((food_uri, FOOD.hasEnergy, Literal(float(row["Energy_kcal"]), datatype=XSD.float)))
    g.add((food_uri, FOOD.hasCarbohydrate, Literal(float(row["Carbohydrate_g"]), datatype=XSD.float)))
    g.add((food_uri, FOOD.isStandalone, Literal(row["Is_Standalone"] == "Yes", datatype=XSD.boolean)))

print(f"Graph built: {len(g)} triples, {df.shape[0]} food nodes")

# ── 4. SAVE THE GRAPH (Turtle format) ───────────
g.serialize(destination="diabetes_prediction/diet_data/food_knowledge_graph.ttl", format="turtle")
print("Saved -> diabetes_prediction/diet_data/food_knowledge_graph.ttl")

# ============================================
# STEP 3: DEMONSTRATION SPARQL QUERIES
# ============================================
print("\n" + "=" * 60)
print("=== SPARQL QUERY DEMONSTRATIONS ===")
print("=" * 60)

# Q1: All Low-GI Protein foods suitable for Breakfast
print("\n--- Q1: Low-GI Protein foods for Breakfast ---")
q1 = """
PREFIX food: <http://example.org/food-ontology#>
SELECT ?label WHERE {
    ?f a food:Food ;
       rdfs:label ?label ;
       food:hasRole food:Protein ;
       food:hasMealType food:Breakfast ;
       food:hasGICategory food:Low .
}
"""
for row in g.query(q1):
    print(f"  - {row.label}")

# Q2: All Moderate-cluster Carb-role foods (NOTE: querying "Diabetic-Safe +
# Carb" together returns ZERO results in this dataset -- a genuine finding,
# not a bug: no raw carbohydrate staple is calorie/GI-light enough to fall
# into the fully "Diabetic-Safe" cluster, confirming the clustering rule's
# strictness. "Moderate" is used below to demonstrate a non-empty query.)
print("\n--- Q2: Moderate-cluster Carb foods ---")
q2 = """
PREFIX food: <http://example.org/food-ontology#>
SELECT ?label ?energy WHERE {
    ?f a food:Food ;
       rdfs:label ?label ;
       food:hasRole food:Carb ;
       food:hasDietCluster food:Moderate ;
       food:hasEnergy ?energy .
}
ORDER BY ?energy
"""
results_q2 = list(g.query(q2))
for row in results_q2[:10]:
    print(f"  - {row.label} ({float(row.energy):.0f} kcal)")
print(f"  ... ({len(results_q2)} total matches)")

# Q3: Count of foods per Food_Role (schema-level summary query)
print("\n--- Q3: Food count per Role ---")
q3 = """
PREFIX food: <http://example.org/food-ontology#>
SELECT ?role (COUNT(?f) AS ?count) WHERE {
    ?f a food:Food ;
       food:hasRole ?role .
}
GROUP BY ?role
ORDER BY DESC(?count)
"""
for row in g.query(q3):
    role_name = str(row.role).split("#")[-1]
    print(f"  - {role_name}: {int(row[1])}")

# Q4: Non-fried, Low-GI Lunch/Dinner Vegetable options (mirrors the exact
# rule used in diet_constraints.py for High-risk patients -- demonstrates
# that the same clinical rule can be expressed declaratively as a query)
print("\n--- Q4: Low-GI Lunch/Dinner Vegetable options, standalone only ---")
q4 = """
PREFIX food: <http://example.org/food-ontology#>
SELECT ?label WHERE {
    ?f a food:Food ;
       rdfs:label ?label ;
       food:hasRole food:Vegetable ;
       food:hasMealType food:Lunch_Dinner ;
       food:hasGICategory food:Low ;
       food:isStandalone true .
    FILTER(!CONTAINS(LCASE(STR(?label)), "fry"))
}
"""
results_q4 = list(g.query(q4))
for row in results_q4[:10]:
    print(f"  - {row.label}")
print(f"  ... ({len(results_q4)} total matches)")

print("\n=== Knowledge Graph + SPARQL demonstration complete! ===")