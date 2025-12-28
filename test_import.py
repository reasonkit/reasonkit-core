import reasonkit
import json
import sys


def test_reasoning():
    print(f"ReasonKit version: {reasonkit.__file__}")

    # Initialize the reasoner with mock mode for testing
    try:
        r = reasonkit.Reasoner(True)
        print("Successfully initialized Reasoner")
    except Exception as e:
        print(f"Failed to initialize Reasoner: {e}")
        sys.exit(1)

    # List available protocols
    protocols = r.list_protocols()
    print(f"Available protocols: {protocols}")

    if "gigathink" not in protocols:
        print("Error: 'gigathink' protocol not found")
        sys.exit(1)

    # Test thinking
    print("\nExecuting 'gigathink' protocol...")
    try:
        result_json = r.think("gigathink", "What is the future of AI agents?")
        result = json.loads(result_json)

        print(f"Execution successful: {result.get('success', False)}")
        print(f"Confidence: {result.get('confidence', 0.0)}")

        if result.get("success"):
            print("\nOutput Data Keys:")
            for key in result.get("data", {}).keys():
                print(f"- {key}")
    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_reasoning()
