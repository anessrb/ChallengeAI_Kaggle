"""
Client Mistral MCP — GeoLifeCLEF Species Predictor

USAGE:
    Terminal 1 (serveur MCP):
        python mcp_server.py

    Terminal 2 (client Mistral):
        export MISTRAL_API_KEY=ta_cle
        python mistral_client.py
"""

import asyncio
import os
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext
from mistralai.extra.mcp.sse import MCPClientSSE, SSEServerParams

MODEL      = "mistral-small-latest"
SERVER_URL = "http://localhost:8080/sse"

async def main():
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("Configure ta cle: export MISTRAL_API_KEY=ta_cle")
        return

    client     = Mistral(api_key)
    mcp_client = MCPClientSSE(sse_params=SSEServerParams(url=SERVER_URL, timeout=100))

    print("=== GeoLifeCLEF Species Assistant (Mistral + MCP) ===")
    print("Exemples:")
    print("  - Quelles especes trouve-t-on a lat=43.5, lon=3.2 ?")
    print("  - Donne moi des infos sur l'espece 1234")
    print("  - Quels surveys sont proches de Lyon (lat=45.75, lon=4.85) ?")
    print("  - Quel est l'environnement a lat=45.0, lon=6.0 ?")
    print("  - quit pour quitter\n")

    async with RunContext(model=MODEL) as run_ctx:
        await run_ctx.register_mcp_client(mcp_client=mcp_client)

        history = []

        while True:
            user_input = input("Vous: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            history.append({"role": "user", "content": user_input})

            run_result = await client.beta.conversations.run_async(
                run_ctx=run_ctx,
                inputs=history,
                system=(
                    "Tu es un assistant specialise en biodiversite vegetale. "
                    "Tu aides a predire et comprendre les especes de plantes presentes "
                    "a differents endroits geographiques grace a un modele de machine learning. "
                    "Utilise les tools disponibles pour repondre precisement."
                ),
            )

            answer = run_result.output_as_text
            history.append({"role": "assistant", "content": answer})
            print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
