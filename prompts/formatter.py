# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, TypedDict, Union, Any

SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]

@dataclass
class ToolUtils(ABC):
    @staticmethod
    @abstractmethod
    def get_function_slots() -> SLOTS: ...

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str: ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]: ...

class DefaultToolUtils(ToolUtils):
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["Action: {{name}}\nAction Input: {{arguments}}\n"]
    
GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具{tool_text}"
)
    
class GLM4ToolUtils(ToolUtils):
    @staticmethod
    def get_function_slots() -> SLOTS:
        return ["{{name}}\n{{arguments}}"]

    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @staticmethod
    def tool_extractor(content: str) -> Union[str, List[Tuple[str, str]]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input)
        except json.JSONDecodeError:
            return content

        return [(tool_name, json.dumps(arguments, ensure_ascii=False))]
    


@dataclass
class Formatter(ABC):
    slots: SLOTS = field(default_factory=list)
    tool_format: Optional[Literal["default", "glm4"]] = None

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS: ...

    def extract(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        raise NotImplementedError


@dataclass
class EmptyFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if has_placeholder:
            raise ValueError("Empty formatter should not contain any placeholder.")

    def apply(self, **kwargs) -> SLOTS:
        return self.slots


@dataclass
class StringFormatter(Formatter):
    def __post_init__(self):
        has_placeholder = False
        for slot in filter(lambda s: isinstance(s, str), self.slots):
            if re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*\}\}", slot):
                has_placeholder = True

        if not has_placeholder:
            raise ValueError("A placeholder is required in the string formatter.")

    def apply(self, **kwargs) -> SLOTS:
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError("Expected a string, got {}".format(value))

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements


@dataclass
class FunctionFormatter(Formatter):
    def __post_init__(self):
        if self.tool_format == "default":
            self.slots = DefaultToolUtils.get_function_slots() + self.slots
        elif self.tool_format == "glm4":
            self.slots = GLM4ToolUtils.get_function_slots() + self.slots
        else:
            raise NotImplementedError("Tool format {} was not found.".format(self.tool_format))

    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        functions: List[Tuple[str, str]] = []
        try:
            tool_calls = json.loads(content)
            if not isinstance(tool_calls, list):  # parallel function call
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                functions.append((tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False)))

        except json.JSONDecodeError:
            functions = []

        elements = []
        for name, arguments in functions:
            for slot in self.slots:
                if isinstance(slot, str):
                    slot = slot.replace("{{name}}", name).replace("{{arguments}}", arguments)
                    elements.append(slot)
                elif isinstance(slot, (dict, set)):
                    elements.append(slot)
                else:
                    raise RuntimeError("Input must be string, set[str] or dict[str, str], got {}".format(type(slot)))

        return elements


@dataclass
class ToolFormatter(Formatter):
    def __post_init__(self):
        if self.tool_format == "default":
            self._tool_formatter = DefaultToolUtils.tool_formatter
            self._tool_extractor = DefaultToolUtils.tool_extractor
        elif self.tool_format == "glm4":
            self._tool_formatter = GLM4ToolUtils.tool_formatter
            self._tool_extractor = GLM4ToolUtils.tool_extractor
        else:
            raise NotImplementedError("Tool format {} was not found.".format(self.tool_format))

    def apply(self, **kwargs) -> SLOTS:
        content = kwargs.pop("content")
        try:
            tools = json.loads(content)
            return [self._tool_formatter(tools) if len(tools) != 0 else ""]
        except json.JSONDecodeError:
            return [""]

    def extract(self, content: str) -> Union[str, List[Tuple[str, str]]]:
        return self._tool_extractor(content)