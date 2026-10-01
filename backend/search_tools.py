from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        seen = set()
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI (deduplicated, with a link back to the material)
            title = course_title
            if lesson_num is not None:
                title += f" - Lesson {lesson_num}"
            if title not in seen:
                seen.add(title)
                link = (
                    self.store.get_lesson_link(course_title, lesson_num)
                    if lesson_num is not None
                    else None
                ) or self.store.get_course_link(course_title)
                sources.append({"title": title, "link": link})
            
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving a course's outline (link and lesson list) from the course catalog"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last outline lookup

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get a course's outline: its title, link, and the complete list of lessons (number and title), each with a content excerpt for summarizing",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_title"]
            }
        }

    def execute(self, course_title: str) -> str:
        """
        Look up the outline for the best-matching course.

        Args:
            course_title: Course title or partial name

        Returns:
            Formatted course outline or error message
        """
        import json

        # Resolve the (possibly partial) name to an exact catalog title
        resolved_title = self.store._resolve_course_name(course_title)
        if not resolved_title:
            return f"No course found matching '{course_title}'"

        try:
            results = self.store.course_catalog.get(ids=[resolved_title])
        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"

        if not results or not results.get('metadatas'):
            return f"No course found matching '{course_title}'"

        metadata = results['metadatas'][0]
        course_link = metadata.get('course_link')
        lessons = json.loads(metadata.get('lessons_json') or "[]")

        lines = [f"Course: {resolved_title}"]
        lines.append(f"Course Link: {course_link or 'N/A'}")
        lines.append(f"Lessons ({len(lessons)}):")
        for lesson in sorted(lessons, key=lambda l: l.get('lesson_number', 0)):
            lesson_number = lesson.get('lesson_number')
            lines.append(f"- Lesson {lesson_number}: {lesson.get('lesson_title')}")
            excerpt = self._get_lesson_excerpt(resolved_title, lesson_number)
            if excerpt:
                lines.append(f"  Excerpt: {excerpt}")

        # Store the course as a source for the UI
        self.last_sources = [{"title": resolved_title, "link": course_link}]

        return "\n".join(lines)

    def _get_lesson_excerpt(self, course_title: str, lesson_number: int, max_chars: int = 600) -> str:
        """Sample the opening and a middle chunk of a lesson so the AI can summarize it"""
        import re

        try:
            results = self.store.course_content.get(
                where={"$and": [
                    {"course_title": course_title},
                    {"lesson_number": lesson_number}
                ]}
            )
        except Exception:
            return ""

        chunks = [
            doc for _, doc in sorted(
                zip(results.get('metadatas') or [], results.get('documents') or []),
                key=lambda pair: pair[0].get('chunk_index', 0)
            )
        ]
        if not chunks:
            return ""

        samples = [chunks[0]]
        if len(chunks) > 2:
            samples.append(chunks[len(chunks) // 2])

        # Drop the "Lesson N content:" / "Course <title> Lesson N content:" prefixes added at ingestion
        samples = [re.sub(r"^(Course .*? )?Lesson \d+ content: ", "", s) for s in samples]
        return " ... ".join(s[:max_chars].replace("\n", " ") for s in samples)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []