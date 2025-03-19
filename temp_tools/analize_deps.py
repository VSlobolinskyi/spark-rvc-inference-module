import os
import ast
from collections import defaultdict


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        # Stores plain "import module" names.
        self.imports = set()
        # Stores "from module import name" mapping.
        self.from_imports = defaultdict(set)
        # Stores names of functions called (this is a best effort approach).
        self.function_calls = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        # node.module can be None in some cases (e.g. relative imports), so we check.
        if node.module:
            for alias in node.names:
                self.from_imports[node.module].add(alias.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name = self.get_call_name(node.func)
        if func_name:
            self.function_calls.add(func_name)
        self.generic_visit(node)

    def get_call_name(self, node):
        """
        Try to resolve the name of the function being called.
        For example:
            - For a simple call like foo(), returns "foo".
            - For a call like module.foo(), returns "module.foo".
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parent = self.get_call_name(node.value)
            if parent:
                return f"{parent}.{node.attr}"
            return node.attr
        return None


def process_file(file_path):
    """
    Parse the Python file at file_path and return an analysis of its imports and function calls.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        tree = ast.parse(file_content, filename=file_path)
    except Exception as e:
        print(f"Skipping {file_path} (could not parse): {e}")
        return None

    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    return {
        "imports": sorted(analyzer.imports),
        "from_imports": {
            module: sorted(names) for module, names in analyzer.from_imports.items()
        },
        "function_calls": sorted(analyzer.function_calls),
    }


def process_directory(root_dir):
    """
    Walk recursively through root_dir, process each .py file, and collect analysis data.
    """
    summary = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                analysis = process_file(file_path)
                if analysis is not None:
                    summary[file_path] = analysis
    return summary


def write_summary(summary, output_file):
    """
    Write the collected analysis to a text file in a human-readable format.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for file, data in summary.items():
            f.write(f"File: {file}\n")
            f.write("  Imports:\n")
            for imp in data["imports"]:
                f.write(f"    - {imp}\n")
            f.write("  From Imports:\n")
            for module, names in data["from_imports"].items():
                f.write(f"    - {module}: {', '.join(names)}\n")
            f.write("  Function Calls:\n")
            for call in data["function_calls"]:
                f.write(f"    - {call}\n")
            f.write("\n")
    print(f"Analysis written to {output_file}")


if __name__ == "__main__":
    project_root = os.getcwd()  # Assumes the script is placed at your project root.
    analysis_summary = process_directory(project_root)
    output_summary_file = "temp_tools/used_dependencies.txt"
    write_summary(analysis_summary, output_summary_file)
