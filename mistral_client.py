"""
Client Mistral qui utilise les tools du serveur MCP GeoLifeCLEF.
Lance mcp_server.py d'abord, puis ce script dans un autre terminal.

Usage:
    python mistral_client.py
"""

import json
import os
from mistralai import Mistral

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")  # export MISTRAL_API_KEY=...
MODEL = "mistral-small-latest"

# ─── IMPORT DES TOOLS LOCALEMENT ──────────────────────────────────────────────
# On importe directement les fonctions du serveur MCP (pas besoin de HTTP)
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from mcp_server import (
    predict_species,
    get_species_info,
    get_environmental_data,
    get_nearby_surveys
)

# ─── DÉFINITION DES TOOLS POUR MISTRAL ────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "predict_species",
            "description": "Prédit les espèces végétales les plus probables à un point GPS en utilisant le modèle XGBoost.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat":   {"type": "number", "description": "Latitude du point GPS"},
                    "lon":   {"type": "number", "description": "Longitude du point GPS"},
                    "top_k": {"type": "integer", "description": "Nombre d'espèces à retourner (défaut: 10)"}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_species_info",
            "description": "Retourne des informations sur une espèce végétale à partir de son ID numérique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "species_id": {"type": "integer", "description": "L'identifiant numérique de l'espèce"}
                },
                "required": ["species_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_environmental_data",
            "description": "Retourne les données environnementales (élévation, bioclimat, sol) au point GPS le plus proche.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_surveys",
            "description": "Retourne les surveys d'observation proches d'un point GPS avec les espèces observées.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat":        {"type": "number",  "description": "Latitude"},
                    "lon":        {"type": "number",  "description": "Longitude"},
                    "radius_km":  {"type": "number",  "description": "Rayon de recherche en km (défaut: 50)"},
                    "max_results":{"type": "integer", "description": "Nombre max de résultats (défaut: 10)"}
                },
                "required": ["lat", "lon"]
            }
        }
    }
]

# ─── DISPATCHER ───────────────────────────────────────────────────────────────
def call_tool(name: str, args: dict) -> str:
    if name == "predict_species":
        return predict_species(**args)
    elif name == "get_species_info":
        return get_species_info(**args)
    elif name == "get_environmental_data":
        return get_environmental_data(**args)
    elif name == "get_nearby_surveys":
        return get_nearby_surveys(**args)
    else:
        return json.dumps({"error": f"Tool {name} non reconnu"})

# ─── BOUCLE DE CONVERSATION ───────────────────────────────────────────────────
def chat(user_message: str, history: list) -> str:
    client = Mistral(api_key=MISTRAL_API_KEY)

    history.append({"role": "user", "content": user_message})

    response = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "Tu es un assistant spécialisé en biodiversité végétale. "
                "Tu aides à prédire et comprendre les espèces de plantes présentes "
                "à différents endroits géographiques grâce à un modèle de machine learning. "
                "Utilise les tools disponibles pour répondre aux questions."
            )}
        ] + history,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    # Gérer les tool calls
    while msg.tool_calls:
        history.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})

        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = call_tool(tc.function.name, args)
            print(f"  [Tool appelé: {tc.function.name}({args})]")
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        response = client.chat.complete(
            model=MODEL,
            messages=[{"role": "system", "content": (
                "Tu es un assistant spécialisé en biodiversité végétale."
            )}] + history,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message

    answer = msg.content or ""
    history.append({"role": "assistant", "content": answer})
    return answer


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("Configure ta clé API: export MISTRAL_API_KEY=ta_cle")
        exit(1)

    print("=== GeoLifeCLEF Species Assistant (Mistral + MCP tools) ===")
    print("Exemples de questions:")
    print("  - Quelles espèces trouve-t-on à lat=43.5, lon=3.2 ?")
    print("  - Donne moi des infos sur l'espèce 1234")
    print("  - Quels surveys sont proches de Paris ?")
    print("  - Quel est l'environnement à lat=45.0, lon=6.0 ?")
    print("  - quit pour quitter\n")

    history = []
    while True:
        user_input = input("Vous: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue
        answer = chat(user_input, history)
        print(f"\nAssistant: {answer}\n")
