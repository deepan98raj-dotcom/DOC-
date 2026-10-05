import os
from typing import Any

import ollama
from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


class VisionAnalysis(BaseModel):
    """Structured visual understanding result."""

    visual_type: str
    summary: str
    detailed_description: str
    text_content: str
    key_elements: list[str]


class VisionService:
    """Analyze document images using a local vision model."""

    def __init__(self):
        self.provider = os.getenv(
            "VISION_PROVIDER",
            "ollama",
        ).lower()

        self.model_name = os.getenv(
            "VISION_MODEL",
            "qwen2.5vl:3b",
        )

        if self.provider != "ollama":
            raise ValueError(
                "Currently supported vision provider is 'ollama'."
            )

    def analyze_image(
        self,
        image_path: str,
        document_id: str,
        page_number: int,
    ) -> dict[str, Any]:
        """Analyze one document image."""

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        prompt = f"""
You are a document vision analysis system.

Analyze the supplied PDF page image.

Document ID: {document_id}
Page number: {page_number}

Classify the visual content as one of:
- chart
- diagram
- figure
- table
- mathematical_visual
- slide
- photo
- other

Then provide:

1. A concise summary.
2. A detailed description.
3. Important visible text.
4. Important labels, axes, values, formulas, arrows,
   shapes, categories, or relationships.

Do not invent information that is not visible.
Use "unknown" when information cannot be determined.

Return valid JSON matching the requested schema.
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
            format=VisionAnalysis.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        content = response.message.content

        try:
            analysis = (
                VisionAnalysis.model_validate_json(
                    content
                )
            )

            result = analysis.model_dump()

        except Exception:
            result = {
                "visual_type": "unknown",
                "summary": content,
                "detailed_description": content,
                "text_content": "",
                "key_elements": [],
            }

        result.update(
            {
                "document_id": document_id,
                "page_number": page_number,
                "image_path": image_path,
                "model": self.model_name,
            }
        )

        return result
