"""
Main runner for the Bedrock Tutorial.

This script allows you to run individual levels or all levels in sequence.

Usage:
    python main.py              # Run interactive menu
    python main.py --level 1    # Run level 1
    python main.py --all        # Run all levels
"""

import sys
import importlib
from pathlib import Path


def run_level(level_number: int) -> bool:
    """
    Run a specific level.

    Args:
        level_number: Level to run (1-10)

    Returns:
        True if successful, False otherwise
    """
    if level_number < 1 or level_number > 10:
        print(f"❌ Invalid level: {level_number}. Choose 1-10.")
        return False

    level_name = f"level_{level_number:02d}_"

    # Find the matching module
    current_dir = Path(__file__).parent
    level_files = list(current_dir.glob(f"{level_name}*.py"))

    if not level_files:
        print(f"❌ Level {level_number} not found.")
        return False

    module_name = level_files[0].stem

    try:
        print(f"\n{'='*80}")
        print(f"Running Level {level_number}: {module_name}")
        print(f"{'='*80}\n")

        # Dynamically import and run the module
        spec = importlib.util.spec_from_file_location(
            module_name,
            level_files[0]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for a Tutorial class and run its demonstrate() method
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and name.endswith("Tutorial"):
                tutorial = obj()
                tutorial.demonstrate()
                return True

        print(f"❌ Could not find tutorial class in {module_name}")
        return False

    except Exception as e:
        print(f"❌ Error running level {level_number}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_levels() -> int:
    """
    Run all levels sequentially.

    Returns:
        Number of successful levels
    """
    successful = 0
    failed = []

    for level in range(1, 11):
        try:
            if run_level(level):
                successful += 1
            else:
                failed.append(level)

            # Pause between levels
            if level < 10:
                input("\n[Press Enter to continue to next level]\n")

        except KeyboardInterrupt:
            print("\n\n⏸️  Tutorial interrupted by user.")
            break

    # Summary
    print(f"\n{'='*80}")
    print(f"TUTORIAL COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Completed: {successful}/10 levels")

    if failed:
        print(f"✗ Failed: {failed}")

    return successful


def interactive_menu():
    """Show interactive menu for level selection."""
    levels = [
        ("Hello World", "Basic connection & invocation"),
        ("Basic Prompting", "System prompts & parameters"),
        ("Prompt Templates", "Reusable prompt patterns"),
        ("Structured Output", "Pydantic & JSON schema"),
        ("Tool Use", "Function calling & agents"),
        ("Conversation Memory", "Multi-turn dialogs"),
        ("Chains", "LangChain composition"),
        ("Vector Store & RAG", "Retrieval-augmented generation"),
        ("Error Handling", "Resilience patterns"),
        ("Streaming & Advanced", "Production patterns"),
    ]

    while True:
        print("\n" + "="*80)
        print("  AWS Bedrock Tutorial - Level Selection")
        print("="*80 + "\n")

        print("Select a level to run:\n")
        for i, (title, description) in enumerate(levels, 1):
            print(f"  {i}  - Level {i}: {title}")
            print(f"      {description}\n")

        print("  A  - Run all levels")
        print("  Q  - Quit")
        print()

        choice = input("Enter your choice (1-10, A, or Q): ").strip().upper()

        if choice == "Q":
            print("\nGoodbye!")
            break

        elif choice == "A":
            run_all_levels()
            break

        elif choice.isdigit() and 1 <= int(choice) <= 10:
            run_level(int(choice))

        else:
            print("❌ Invalid choice. Please try again.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--level" and len(sys.argv) > 2:
            level = int(sys.argv[2])
            run_level(level)

        elif sys.argv[1] == "--all":
            run_all_levels()

        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)

        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print(__doc__)

    else:
        # Show interactive menu
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTutorial interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
