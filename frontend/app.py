from pathlib import Path
from urllib.parse import quote

import requests
import streamlit as st


st.set_page_config(
    page_title="DocMind AI Studio",
    page_icon="D",
    layout="wide",
)


API_BASE_URL = st.sidebar.text_input(
    "FastAPI URL",
    "http://127.0.0.1:8000",
)


st.title("DocMind AI Studio")
st.caption(
    "Multimodal document intelligence and RAG assistant"
)


def get_documents():
    response = requests.get(
        f"{API_BASE_URL}/documents",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


with st.sidebar:
    st.header("Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
    )

    if st.button(
        "Upload & Index",
        use_container_width=True,
    ):
        if uploaded_file is None:
            st.warning(
                "Please select a PDF first."
            )
        else:
            try:
                with st.spinner(
                    "Extracting, indexing and processing multimodal assets..."
                ):
                    response = requests.post(
                        f"{API_BASE_URL}/documents/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=600,
                    )

                response.raise_for_status()

                result = response.json()

                st.success(
                    f"Text chunks: "
                    f"{result['indexed_chunk_count']} | "
                    f"Multimodal assets: "
                    f"{result['indexed_multimodal_count']}"
                )

                st.session_state[
                    "selected_document_id"
                ] = result["document_id"]

                st.rerun()

            except requests.RequestException as exc:
                st.error(
                    f"Upload failed: {exc}"
                )


st.header("Document")

try:
    documents = get_documents()

except requests.RequestException:
    st.error(
        "Could not connect to FastAPI. "
        "Make sure Uvicorn is running."
    )
    st.stop()


if not documents:
    st.info(
        "No indexed documents available."
    )
    st.stop()


document_labels = {
    (
        f"{doc['filename']} "
        f"({doc['chunk_count']} chunks)"
    ): doc["document_id"]
    for doc in documents
}

labels = list(
    document_labels.keys()
)

default_index = 0

saved_document_id = st.session_state.get(
    "selected_document_id"
)

if saved_document_id:
    for index, doc in enumerate(documents):
        if doc["document_id"] == saved_document_id:
            default_index = index
            break


selected_label = st.selectbox(
    "Select a document",
    labels,
    index=default_index,
)

selected_document_id = document_labels[
    selected_label
]

st.caption(
    f"Document ID: {selected_document_id}"
)

st.divider()

st.header("Ask your document")

query = st.text_area(
    "Question",
    placeholder=(
        "Example: Explain the higher-order tensor "
        "example involving matrix A and vector x."
    ),
    height=100,
)


if st.button(
    "Ask DocMind",
    type="primary",
    use_container_width=True,
):

    if not query.strip():
        st.warning(
            "Please enter a question."
        )
        st.stop()

    encoded_document_id = quote(
        selected_document_id,
        safe="",
    )

    try:
        with st.spinner(
            "Searching, reranking and generating answer..."
        ):
            response = requests.post(
                f"{API_BASE_URL}/documents/"
                f"{encoded_document_id}/query",
                json={
                    "query": query,
                    "top_k": 5,
                    "candidate_k": 20,
                },
                timeout=300,
            )

        response.raise_for_status()

        result = response.json()

        st.subheader("Answer")

        st.write(
            result["answer"]
        )

        st.divider()

        st.subheader(
            f"Sources ({len(result['sources'])})"
        )

        for source in result["sources"]:

            source_id = source.get(
                "source_id",
                "Unknown",
            )

            page_number = source.get(
                "page_number",
                "Unknown",
            )

            source_type = source.get(
                "source_type",
                source.get(
                    "content_type",
                    "text",
                ),
            )

            content_type = source.get(
                "content_type",
                "text",
            )

            reranker_score = source.get(
                "reranker_score"
            )

            with st.expander(
                f"{source_id} | "
                f"{source_type} | "
                f"Page {page_number}",
                expanded=(
                    source_type == "vision"
                    and source_id == "S1"
                ),
            ):

                st.write(
                    f"**Document:** "
                    f"{source['filename']}"
                )

                st.write(
                    f"**Page:** "
                    f"{page_number}"
                )

                st.write(
                    f"**Content type:** "
                    f"{content_type}"
                )

                st.write(
                    f"**Source type:** "
                    f"{source_type}"
                )

                st.write(
                    f"**Chunk / Asset ID:** "
                    f"{source['chunk_id']}"
                )

                if reranker_score is not None:
                    st.write(
                        f"**Reranker score:** "
                        f"{reranker_score:.4f}"
                    )

                asset_path = source.get(
                    "asset_path"
                )

                if asset_path:
                    image_path = Path(
                        asset_path
                    )

                    if image_path.exists():
                        st.write(
                            "**Visual Evidence:**"
                        )

                        st.image(
                            str(image_path),
                            caption=(
                                f"Page {page_number} "
                                f"visual evidence"
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.warning(
                            "Visual asset was indexed "
                            "but the image file could not "
                            f"be found: {asset_path}"
                        )


    except requests.HTTPError:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = "Unknown API error."

        st.error(
            f"Query failed: {error_detail}"
        )

    except requests.RequestException as exc:
        st.error(
            f"Could not reach FastAPI: {exc}"
        )

    except Exception as exc:
        st.error(
            f"Unexpected application error: {exc}"
        )
