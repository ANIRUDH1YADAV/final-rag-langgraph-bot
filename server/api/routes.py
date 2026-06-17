from fastapi import APIRouter, UploadFile, File, Form

from config.settings import MODEL_OPTIONS
from core.vector_database import (
    get_collections_count,
    find_similar_chunks,
    upsert_vectorstore_from_pdfs
)

from core.langgraph.graph import graph

from api.schemas import (
    SearchQueryRequest,
    ChatRequest,
    StandardAPIResponse
)

from utils.logger import logger

router = APIRouter()
chat_memory = []


@router.get("/health", response_model=StandardAPIResponse)
def health_check():
    logger.debug("Health check requested")

    return StandardAPIResponse(
        status="success",
        data="ok",
        message="Service is healthy"
    )


@router.get("/llm", response_model=StandardAPIResponse)
async def get_llm_options():

    logger.debug("Fetching LLM providers.")

    return StandardAPIResponse(
        status="success",
        data=[
            provider.title()
            for provider in MODEL_OPTIONS.keys()
        ]
    )


@router.get("/llm/{model_provider}", response_model=StandardAPIResponse)
async def get_llm_models(model_provider: str):

    model_provider = model_provider.lower()

    if model_provider not in MODEL_OPTIONS:

        logger.warning(
            f"Invalid model provider: {model_provider}"
        )

        return StandardAPIResponse(
            status="error",
            message="Invalid model provider."
        )

    logger.debug(
        f"Fetching models for provider: {model_provider}"
    )

    return StandardAPIResponse(
        status="success",
        data=MODEL_OPTIONS[model_provider]["models"]
    )


@router.post(
    "/upload_and_process_pdfs",
    response_model=StandardAPIResponse
)
async def upload_and_process_pdfs(
    files: list[UploadFile] = File(...),
    model_provider: str = Form(...)
):

    try:

        model_provider = model_provider.lower()

        logger.info(
            f"Received {len(files)} files "
            f"for model provider: {model_provider}"
        )

        await upsert_vectorstore_from_pdfs(
            files,
            model_provider
        )

        logger.info(
            "Files processed successfully"
        )

        return StandardAPIResponse(
            status="success",
            data="PDFs processed successfully."
        )

    except Exception as e:

        logger.exception(
            "Error while uploading and processing files"
        )

        return StandardAPIResponse(
            status="error",
            message=str(e)
        )


@router.get(
    "/vector_store/count/{model_provider}",
    response_model=StandardAPIResponse
)
async def get_vectorstore_count(
    model_provider: str
):

    try:

        model_provider = model_provider.lower()

        logger.info(
            f"Getting collection count for provider: "
            f"{model_provider}"
        )

        count = get_collections_count(
            model_provider
        )

        return StandardAPIResponse(
            status="success",
            data=count
        )

    except Exception as e:

        logger.exception(
            "Error getting collection count"
        )

        return StandardAPIResponse(
            status="error",
            message=str(e)
        )


@router.post(
    "/vector_store/search",
    response_model=StandardAPIResponse
)
async def get_vectorstore_search(
    request: SearchQueryRequest
):

    try:

        model_provider = request.model_provider.lower()

        logger.info(
            f"Search requested with query: "
            f"{request.query}"
        )

        results = find_similar_chunks(
            model_provider,
            request.query
        )

        return StandardAPIResponse(
            status="success",
            data=results
        )

    except Exception as e:

        logger.exception(
            "Error during similarity search"
        )

        return StandardAPIResponse(
            status="error",
            message=str(e)
        )


@router.post("/chat", response_model=StandardAPIResponse)
async def chat(request: ChatRequest):

    global chat_memory

    try:

        message = request.message

        model_name = request.model_name

        model_provider = (
            request.model_provider.lower()
        )

        logger.debug(
            f"Chat request for model: "
            f"{model_name} "
            f"(provider: {model_provider})"
        )

        if model_provider not in MODEL_OPTIONS:

            logger.warning(
                "Invalid model provider."
            )

            return StandardAPIResponse(
                status="error",
                message="Invalid model provider."
            )

        if model_name not in MODEL_OPTIONS[
            model_provider
        ]["models"]:

            logger.warning(
                "Invalid model name."
            )

            return StandardAPIResponse(
                status="error",
                message="Invalid model name."
            )

        result = graph.invoke(
            {
                "question": message,
                "chat_history": chat_memory,
                "model_provider": model_provider,
                "model_name": model_name
            }
        )

        response = result["answer"]

        chat_memory = result.get(
            "chat_history",
            chat_memory
        )

        print(
        "\nCURRENT MEMORY:\n",
        chat_memory
      )

        logger.debug(
            "Chat response generated successfully"
        )

        return StandardAPIResponse(
            status="success",
            data=response
        )

    except Exception as e:

        logger.exception(
            "Chat endpoint encountered an error"
        )

        return StandardAPIResponse(
            status="error",
            message=str(e)
        )
    


