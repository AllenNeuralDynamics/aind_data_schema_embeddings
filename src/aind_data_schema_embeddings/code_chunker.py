"""Python code chunker that preserves code structure and syntax"""

import ast
import json
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class CodeChunk:
    """Class attributes for each code chunk"""

    content: str  # code content
    type: str  # 'import', 'class', 'function', 'method'
    name: str  # name of class/function etc
    file_name: str
    parent: Optional[str] = None
    docstring: Optional[str] = None


class PythonCodeChunker:
    """Code chunker class"""

    def __init__(self, file_path: str, file_name: str):
        """Constructor"""

        with open(file_path, "r") as file:
            self.content = file.read()
        self.tree = ast.parse(self.content)
        self.chunks: List[CodeChunk] = []
        self.current_class = None
        self.max_chunk_size = 8192
        self.file_name = file_name

    def extract_docstring(self, node) -> str:
        """Extract docstring from an AST node."""

        if ast.get_docstring(node):
            return ast.get_docstring(node)
        return ""

    def process_imports(self):
        """Extract and chunk import statements."""

        import_chunk = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_str = ast.get_source_segment(self.content, node)
                if import_str:
                    import_chunk.append(import_str)

        if import_chunk:
            self.chunks.append(
                CodeChunk(
                    content="\n".join(import_chunk),
                    type="import",
                    name="imports",
                    docstring="Module imports",
                    file_name=self.file_name,
                )
            )

    def process_classes(self):
        """Process class definitions, handling large classes appropriately."""

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(self.content, node)

                if len(class_code) > self.max_chunk_size:
                    # Handle large class
                    class_chunks = self.split_large_class(node)
                    self.chunks.extend(class_chunks)
                else:
                    # Handle normal-sized class
                    self.chunks.append(
                        CodeChunk(
                            content=class_code,
                            type="class_definition",
                            name=node.name,
                            docstring=self.extract_docstring(node),
                            file_name=self.file_name,
                        )
                    )

    def split_large_class(self, node: ast.ClassDef) -> List[CodeChunk]:
        """Split a large class into multiple manageable chunks."""

        chunks = []

        # First chunk: Class definition and docstring
        class_header = node.name
        docstring = self.extract_docstring(node)

        chunks.append(
            CodeChunk(
                content=f"{class_header}\n \
                    {textwrap.indent(docstring, '    ') if docstring else ''}",
                type="class_definition",
                name=node.name,
                docstring=docstring,
                file_name=self.file_name,
            )
        )

        # Collect class attributes
        attributes = []
        methods = []

        for child in node.body:
            if isinstance(child, ast.AnnAssign) or isinstance(
                child, ast.Assign
            ):
                attributes.append(ast.get_source_segment(self.content, child))
            elif isinstance(child, ast.FunctionDef):
                methods.append(child)

        # Create chunk for class attributes if any
        attribute_chunks = self._process_attributes(attributes, node.name)
        chunks.extend(attribute_chunks)

        # Process methods
        method_chunks = self._process_methods(methods, node.name)
        chunks.extend(method_chunks)

        return chunks

    def _process_attributes(
        self,
        attributes: List[Union[ast.AnnAssign, ast.Assign]],
        class_name: str,
    ) -> List[CodeChunk]:
        """Process class methods, potentially splitting large methods."""

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for attribute in attributes:
            # method_code = ast.get_source_segment(self.content, method)
            attribute_size = len(attribute)

            if current_size + attribute_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            type="class_attributes",
                            name=f"{class_name}_attributes_part_{chunk_index}",
                            parent=class_name,
                            file_name=self.file_name,
                        )
                    )
                    current_chunk = []
                    current_size = 0
                    chunk_index += 1

            current_chunk.append(attribute)
            current_size += attribute_size

        if current_chunk:
            if chunk_index > 0:
                chunks.append(
                    CodeChunk(
                        content="\n".join(current_chunk),
                        type="class_attributes",
                        name=f"{class_name}_attributes_part_{chunk_index}",
                        parent=class_name,
                        file_name=self.file_name,
                    )
                )
            else:
                chunks.append(
                    CodeChunk(
                        content="\n".join(current_chunk),
                        type="class_attributes",
                        name=f"{class_name}_attributes",
                        parent=class_name,
                        file_name=self.file_name,
                    )
                )

        return chunks

    def _process_methods(
        self, methods: List[ast.FunctionDef], class_name: str
    ) -> List[CodeChunk]:
        """Process class methods, potentially splitting large methods."""

        chunks = []
        current_chunk = []
        current_size = 0

        for method in methods:
            method_code = ast.get_source_segment(self.content, method)
            method_size = len(method_code)

            # if method_size > self.max_chunk_size:
            #     # Handle large individual methods
            #     method_chunks = self._split_large_method(method, class_name)
            #     chunks.extend(method_chunks)
            # else:
            if current_size + method_size > self.max_chunk_size:
                # Create a new chunk with accumulated methods
                if current_chunk:
                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            type="class_method",
                            name=f"{method.name}",
                            parent=class_name,
                            file_name=self.file_name,
                        )
                    )
                    current_chunk = []
                    current_size = 0
            current_chunk.append(method_code)
            current_size += method_size

        # Add remaining methods
        if current_chunk:
            chunks.append(
                CodeChunk(
                    content="\n".join(current_chunk),
                    type="class_method",
                    name=f"{method.name}",
                    parent=class_name,
                    file_name=self.file_name,
                )
            )

        return chunks

    def process_standalone_functions(self):
        """Process top-level function definitions."""

        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.FunctionDef):
                func_code = ast.get_source_segment(self.content, node)
                docstring = self.extract_docstring(node)

                chunk = CodeChunk(
                    content=func_code,
                    type="function",
                    name=node.name,
                    docstring=docstring,
                    file_name=self.file_name,
                )
                self.chunks.append(chunk)

    def create_chunks(self) -> List[CodeChunk]:
        """Create all chunks from the Python file."""

        self.chunks = []
        self.process_imports()
        self.process_classes()
        self.process_standalone_functions()

        combined_chunk_list = []
        curr_chunk = ""
        chunk_size = 0

        for chunk in self.chunks:
            #     print(type(chunk))
            chunk_str = json.dumps(chunk.__dict__)

            chunk_size = len(curr_chunk)
            chunk_to_add_size = len(chunk_str)

            if chunk_size + chunk_to_add_size >= self.max_chunk_size:
                combined_chunk_list.append(curr_chunk)
                curr_chunk = chunk_str
            else:
                curr_chunk += chunk_str

        # Append the last chunk if it has any content
        if curr_chunk:
            combined_chunk_list.append(curr_chunk)

        return combined_chunk_list
