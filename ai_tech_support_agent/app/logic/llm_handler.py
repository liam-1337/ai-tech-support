import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional, Tuple

# Assuming app.config is accessible and correctly set up
# This will be the case if the application is run via FastAPI main.py or scripts that set up sys.path
from app import config

# Get a logger for this module
logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    Manages the loading of a Hugging Face Language Model (LLM) and tokenizer,
    formats prompts, and generates answers based on context and a query.
    Uses class methods for resource management to ensure model is loaded once.
    """
    _model: Optional[PreTrainedModel] = None
    _tokenizer: Optional[PreTrainedTokenizer] = None
    _model_name_loaded: Optional[str] = None  # To track which model name is currently loaded
    _device: str = "cpu"  # Default device

    @classmethod
    def _initialize(cls, model_name_to_load: str) -> None:
        """
        Initializes the LLM model and tokenizer if they haven't been loaded yet
        or if the requested model name has changed.

        Args:
            model_name_to_load: The Hugging Face model name from the configuration.
        """
        if cls._model and cls._tokenizer and cls._model_name_loaded == model_name_to_load:
            logger.info(f"LLM '{cls._model_name_loaded}' and tokenizer already initialized on device '{cls._device}'.")
            return

        cls._model_name_loaded = model_name_to_load
        cls._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing LLM '{cls._model_name_loaded}' on device '{cls._device}'.")

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name_loaded)

            # device_map="auto" requires 'accelerate' and works well for multi-GPU or large models.
            # If only CPU, it maps to CPU. If single GPU, maps to that GPU.
            # Forcing CPU: device_map="cpu"
            model_kwargs = {}
            if cls._device == "cuda":
                if not torch.cuda.is_available(): # Should not happen if _device is cuda, but double check
                     logger.warning("CUDA specified but not available. Forcing LLM to CPU.")
                     cls._device = "cpu" # Fallback to CPU
                     model_kwargs['device_map'] = "cpu"
                else:
                    model_kwargs['device_map'] = "auto" # Handles multi-GPU, single-GPU
            else: # CPU
                model_kwargs['device_map'] = "cpu"

            cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name_loaded, **model_kwargs)

            # If not using device_map or want explicit control for single device:
            # cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name_loaded).to(cls._device)

            logger.info(f"LLM '{cls._model_name_loaded}' and tokenizer loaded successfully on effective device: {cls._model.device}.")
        except Exception as e:
            logger.error(f"Failed to load LLM model or tokenizer '{cls._model_name_loaded}': {e}", exc_info=True)
            cls._model = None  # Ensure partial loads are reset
            cls._tokenizer = None
            cls._model_name_loaded = None
            raise RuntimeError(f"Failed to initialize LLM resources for {model_name_to_load}.") from e

    @classmethod
    def get_llm_resources(cls) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Ensures the LLM model and tokenizer are initialized (using the name from config)
        and returns them.

        Returns:
            A tuple containing the initialized model and tokenizer.

        Raises:
            RuntimeError: If the model or tokenizer could not be initialized.
        """
        model_name_from_config = config.LLM_MODEL_NAME # Get current config for initialization
        if cls._model is None or cls._tokenizer is None or cls._model_name_loaded != model_name_from_config:
            cls._initialize(model_name_from_config)

        if cls._model is None or cls._tokenizer is None: # Check again after trying to initialize
            raise RuntimeError("LLM model and/or tokenizer are not available after initialization attempt.")
        return cls._model, cls._tokenizer

    @staticmethod
    def format_prompt(user_query: str, context_chunks: List[str], prompt_template: str) -> str:
        """
        Formats the prompt for the LLM using a template, user query, and context chunks.

        Args:
            user_query: The user's question.
            context_chunks: A list of text snippets to be used as context.
            prompt_template: The template string with placeholders for '{context_chunks}' and '{question}'.

        Returns:
            The formatted prompt string.
        """
        if not context_chunks:
            logger.warning("Formatting prompt with no context chunks provided.")
            # Adapt template if context is optional or provide a specific no-context instruction
            # For now, we'll just join them, resulting in an empty string for context.
            # A more sophisticated approach might alter the template or instructions.
            formatted_context = "No specific context provided."
        else:
            formatted_context = "\n---\n".join(context_chunks) # Join chunks with a separator

        try:
            final_prompt = prompt_template.format(context_chunks=formatted_context, question=user_query)
            logger.debug(f"Formatted LLM Prompt (first 200 chars): {final_prompt[:200]}...")
            return final_prompt
        except KeyError as e:
            logger.error(f"Failed to format prompt. Missing key in template: {e}. Template: '{prompt_template}'")
            # Fallback or raise error
            return f"Question: {user_query}\nContext: {formatted_context}\n(Error: Prompt template formatting failed)"


    @classmethod
    def generate_answer(cls, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer using the LLM based on the query and context.

        Args:
            query: The user's question.
            context_chunks: A list of context chunks relevant to the query.

        Returns:
            The generated answer string, or an error message if generation fails.
        """
        try:
            model, tokenizer = cls.get_llm_resources()
        except RuntimeError as e:
            logger.error(f"Cannot generate answer due to LLM resource initialization failure: {e}")
            return "Error: LLM model is not available or failed to load. Please check server logs."

        prompt_text = LLMGenerator.format_prompt(query, context_chunks, config.LLM_PROMPT_TEMPLATE)

        # Tokenize the prompt. `truncation=True` and `max_length` are important for long prompts.
        # The tokenizer's max_length might differ from the model's generation max_new_tokens.
        # 512 or 1024 are common tokenizer max_lengths for input.
        try:
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 512
            )
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {e}", exc_info=True)
            return "Error: Could not process the input for the language model."

        # Move input tensors to the same device as the model.
        # This is crucial if the model is on a specific GPU.
        # For device_map="auto", model.device gives the primary device where generation occurs or where inputs should go.
        try:
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
        except Exception as e:
            logger.error(f"Error moving input tensors to model device '{model.device}': {e}", exc_info=True)
            return "Error: Internal setup error for language model processing."

        logger.info(f"Generating LLM answer. Prompt length: {len(input_ids[0])} tokens. Max new tokens: {config.LLM_MAX_NEW_TOKENS}.")
        try:
            with torch.no_grad(): # Ensure no gradients are computed during inference
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                    temperature=float(config.LLM_TEMPERATURE), # Ensure temperature is float
                    # Other common parameters:
                    # num_beams=config.LLM_NUM_BEAMS, # If using beam search
                    # top_k=config.LLM_TOP_K,
                    # top_p=config.LLM_TOP_P,
                    # do_sample=True, # Important for temperature to have an effect
                    # early_stopping=True, # if num_beams > 1
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Successfully generated LLM Answer (snippet): {answer[:150].replace(chr(10), ' ')}...") # Replace newlines for cleaner log snippet
            return answer.strip()
        except Exception as e:
            logger.error(f"Error during LLM answer generation with model '{cls._model_name_loaded}': {e}", exc_info=True)
            return "Error: The language model encountered an issue while generating an answer."

if __name__ == '__main__':
    # This block is for basic standalone testing of the LLMGenerator.
    # It requires app.config to be loadable, so ensure Python path is correct or run from project root.

    # Setup basic logging for testing this module directly
    logging.basicConfig(
        level=logging.DEBUG, # Use DEBUG to see more detailed logs from LLMGenerator
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("Running LLMHandler module directly for testing...")

    # --- Test Case 1: Basic Prompt Formatting ---
    test_query = "What is FAISS?"
    test_context = [
        "FAISS is a library for efficient similarity search.",
        "It was developed by Facebook AI Research (FAIR).",
        "FAISS allows searching in sets of vectors of high dimensionality."
    ]
    expected_prompt_start = "Use the following pieces of context to answer the question at the end."

    formatted_prompt = LLMGenerator.format_prompt(test_query, test_context, config.LLM_PROMPT_TEMPLATE)
    logger.info(f"\n--- Test Case 1: Formatted Prompt ---\n{formatted_prompt}\n---------------------------------")
    assert expected_prompt_start in formatted_prompt
    assert test_query in formatted_prompt
    assert test_context[0] in formatted_prompt
    logger.info("Test Case 1 (Prompt Formatting) Passed.")

    # --- Test Case 2: Prompt Formatting with No Context ---
    formatted_prompt_no_context = LLMGenerator.format_prompt(test_query, [], config.LLM_PROMPT_TEMPLATE)
    logger.info(f"\n--- Test Case 2: Formatted Prompt (No Context) ---\n{formatted_prompt_no_context}\n---------------------------------")
    assert "No specific context provided." in formatted_prompt_no_context
    logger.info("Test Case 2 (Prompt Formatting - No Context) Passed.")

    # --- Test Case 3: Answer Generation (requires model download on first run) ---
    # This test will use the model specified in your .env or app/config.py (e.g., "google/flan-t5-base")
    # It might take time if the model needs to be downloaded.
    # Ensure you have internet access and sufficient disk space/RAM.
    logger.info(f"\n--- Test Case 3: Answer Generation (using model: {config.LLM_MODEL_NAME}) ---")
    logger.warning("This test may download the LLM model if not cached, which can take time and resources.")

    # Forcing a specific small model for CI/testing if needed, otherwise uses config
    # original_llm_model_name = config.LLM_MODEL_NAME
    # config.LLM_MODEL_NAME = "distilgpt2" # A smaller model for faster testing
    # logger.info(f"Temporarily switched to {config.LLM_MODEL_NAME} for testing generate_answer.")

    try:
        # Test with context
        answer_with_context = LLMGenerator.generate_answer(test_query, test_context)
        logger.info(f"Answer (with context) for '{test_query}':\n{answer_with_context}")
        assert answer_with_context and not answer_with_context.startswith("Error:"), "Answer generation with context failed or returned an error."

        # Test with no context (model should ideally say it doesn't know or answer generally)
        answer_no_context = LLMGenerator.generate_answer(test_query, [])
        logger.info(f"Answer (no context) for '{test_query}':\n{answer_no_context}")
        assert answer_no_context and not answer_no_context.startswith("Error:"), "Answer generation with no context failed or returned an error."

        logger.info("Test Case 3 (Answer Generation) Passed (check output for quality).")

    except Exception as e:
        logger.error(f"Test Case 3 (Answer Generation) Failed: {e}", exc_info=True)
        logger.error("Ensure you have 'accelerate' installed (`pip install accelerate`) and sufficient resources.")
    finally:
        # config.LLM_MODEL_NAME = original_llm_model_name # Restore original model name if changed
        pass

    logger.info("LLMHandler module test run complete.")
