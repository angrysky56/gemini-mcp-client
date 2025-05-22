"""
Example usage of the MCP Gemini Client with model selection
"""

import asyncio
from mcp_gemini_client import MCPClient


async def main():
    """Example usage of the MCP Client with different models."""

    # Example 1: Initialize with default model
    print("🚀 Example 1: Default Model")
    client = MCPClient()

    try:
        server_path = "examples/echo_server.py"

        print("🔌 Connecting to MCP server...")
        await client.connect_to_server(server_path)

        # Get server information
        print("\n📋 Server Information:")
        server_info = await client.get_server_info()
        print(f"  Current Model: {server_info['model']}")
        print(f"  Available Models: {len(server_info['available_models'])}")
        print(f"  Tools: {len(server_info['tools'])}")

        # Example interaction
        response = await client.get_response("What tools are available?")
        print(f"\nResponse: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()

    print("\n" + "="*60 + "\n")

    # Example 2: Initialize with specific model
    print("🚀 Example 2: Specific Model")
    client2 = MCPClient(model="gemini-2.5-pro-preview-03-25")

    try:
        await client2.connect_to_server("examples/echo_server.py")

        server_info = await client2.get_server_info()
        print(f"  Current Model: {server_info['model']}")

        # Change model during runtime
        print("\n🔄 Changing model to gemini-1.5-flash...")
        client2.set_model("gemini-1.5-flash")

        updated_info = await client2.get_server_info()
        print(f"  Updated Model: {updated_info['model']}")

        # Test with new model
        response = await client2.get_response("Hello! What model are you using?")
        print(f"\nResponse: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client2.close()

    print("\n" + "="*60 + "\n")

    # Example 3: Model comparison
    print("🚀 Example 3: Model Comparison")

    models_to_test = ["gemini-2.0-flash", "gemini-1.5-pro"]
    question = "Explain what an MCP server does in one sentence."

    for model in models_to_test:
        print(f"\n🤖 Testing {model}:")
        client = MCPClient(model=model)
        try:
            await client.connect_to_server("examples/echo_server.py")
            response = await client.get_response(question)
            print(f"  Response: {response}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        finally:
            await client.close()


async def interactive_model_selection():
    """Interactive example showing model selection."""
    print("\n🎮 Interactive Model Selection")
    print("=" * 40)

    client = MCPClient()
    available_models = client.get_available_models()

    print("Available models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    try:
        choice = input(f"\nSelect model (1-{len(available_models)}): ")
        selected_model = available_models[int(choice) - 1]

        print(f"Selected: {selected_model}")
        client.set_model(selected_model)

        # Connect and test
        await client.connect_to_server("examples/echo_server.py")

        question = input("Ask a question: ")
        response = await client.get_response(question)
        print(f"\nResponse: {response}")

    except (ValueError, IndexError):
        print("❌ Invalid selection")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    print("🎯 MCP Gemini Client - Model Selection Examples")
    print("=" * 60)

    # Run basic examples
    asyncio.run(main())

    # Uncomment for interactive example
    # asyncio.run(interactive_model_selection())
