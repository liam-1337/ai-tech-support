import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional, Tuple, Dict, Any
import re

from app import config
from app.logic.common_automations import get_password_reset_guidance, get_connectivity_check_guidance

logger = logging.getLogger(__name__)

PASSWORD_RESET_KEYWORDS = ["reset password", "forgot password", "change password", "password help", "unlock account", "account locked", "recover password"]
PASSWORD_RESPONSE_KEYWORDS = ["password", "reset", "account", "login", "credentials", "unlock", "recover", "sign in", "log in"]
CONNECTIVITY_KEYWORDS = [
    "can't connect", "cannot connect", "no internet", "wifi problem", "wi-fi problem",
    "network down", "internet down", "connection issue", "connectivity issue",
    "internet not working", "wifi not working", "network error", "no connection"
]
CONNECTIVITY_RESPONSE_KEYWORDS = [
    "connect", "internet", "wifi", "wi-fi", "network", "connection", "online",
    "router", "modem", "signal", "cable", "ethernet", "ip address", "ping"
]

class LLMGenerator:
    _model: Optional[PreTrainedModel] = None
    _tokenizer: Optional[PreTrainedTokenizer] = None
    _model_name_loaded: Optional[str] = None
    _device: str = "cpu"

    @staticmethod
    def _format_structured_guidance(guidance_data: Dict[str, Any]) -> str:
        if not guidance_data or not guidance_data.get("steps"):
            logger.warning("Attempted to format empty or invalid guidance data.")
            return ""

        formatted_guidance = f"\n\n---\n**Automated Guide: {guidance_data['title']}**\n\n"
        if guidance_data.get('steps'):
            formatted_guidance += "**Steps:**\n"
            for i, step in enumerate(guidance_data['steps']):
                formatted_guidance += f"{i+1}. {step}\n"
        if guidance_data.get('notes'):
            formatted_guidance += "\n**Important Notes:**\n"
            for note in guidance_data['notes']:
                formatted_guidance += f"- {note}\n"
        return formatted_guidance

    @classmethod
    def _initialize(cls, model_name_to_load: str) -> None:
        if cls._model and cls._tokenizer and cls._model_name_loaded == model_name_to_load:
            logger.info(f"LLM '{cls._model_name_loaded}' and tokenizer already initialized on device '{cls._device}'.")
            return
        cls._model_name_loaded = model_name_to_load
        cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing LLM '{cls._model_name_loaded}' on device '{cls._device}'.")
        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name_loaded)
            model_kwargs = {}
            if cls._device == "cuda":
                if not torch.cuda.is_available():
                     logger.warning("CUDA specified but not available. Forcing LLM to CPU.")
                     cls._device = "cpu"
                     model_kwargs['device_map'] = "cpu"
                else: model_kwargs['device_map'] = "auto"
            else: model_kwargs['device_map'] = "cpu"
            cls._model = AutoModelForSeq2SeqLM.from_pretrained(cls._model_name_loaded, **model_kwargs)
            logger.info(f"LLM '{cls._model_name_loaded}' and tokenizer loaded successfully on effective device: {cls._model.device}.")
        except Exception as e:
            logger.error(f"Failed to load LLM model or tokenizer '{cls._model_name_loaded}': {e}", exc_info=True)
            cls._model, cls._tokenizer, cls._model_name_loaded = None, None, None
            raise RuntimeError(f"Failed to initialize LLM resources for {model_name_to_load}.") from e

    @classmethod
    def get_llm_resources(cls) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_name_from_config = config.LLM_MODEL_NAME
        if cls._model is None or cls._tokenizer is None or cls._model_name_loaded != model_name_from_config:
            cls._initialize(model_name_from_config)
        if cls._model is None or cls._tokenizer is None:
            raise RuntimeError("LLM model and/or tokenizer are not available after initialization attempt.")
        return cls._model, cls._tokenizer

    @staticmethod
    def format_prompt(
        user_query: str, context_chunks: List[str],
        prompt_template: str, conversation_history: Optional[List[str]] = None
    ) -> str:
        formatted_context = "\n---\n".join(context_chunks) if context_chunks else "No specific context provided for this query."
        if not conversation_history:
            formatted_history = "No prior conversation history available for this query."
        else:
            formatted_history = "\n".join([f"Turn {i+1}: {turn}" for i, turn in enumerate(reversed(conversation_history))])
        try:
            final_prompt = prompt_template.format(
                context_chunks=formatted_context, question=user_query, conversation_history=formatted_history
            )
            logger.debug(f"Formatted LLM Prompt (first 300 chars): {final_prompt[:300]}...")
            return final_prompt
        except KeyError as e:
            logger.error(f"Failed to format prompt. Missing key: {e}. Template: '{prompt_template}'")
            return f"System Error: Prompt template formatting failed. Q: {user_query} Cxt: {formatted_context} Hist: {formatted_history}"

    @classmethod
    def generate_answer(cls, query: str, context_chunks: List[str], conversation_history: Optional[List[str]] = None) -> str:
        try:
            model, tokenizer = cls.get_llm_resources()
        except RuntimeError as e:
            logger.error(f"Cannot generate answer due to LLM init failure: {e}")
            return "Error: LLM model not available. Check server logs."

        prompt_text = LLMGenerator.format_prompt(
            query, context_chunks, config.LLM_PROMPT_TEMPLATE, conversation_history
        )
        try:
            inputs = tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 512
            )
        except Exception as e:
            logger.error(f"Error tokenizing prompt: {e}", exc_info=True)
            return "Error: Could not process input for LLM."
        try:
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
        except Exception as e:
            logger.error(f"Error moving tensors to model device '{model.device}': {e}", exc_info=True)
            return "Error: Internal setup error for LLM processing."

        logger.info(f"Generating LLM answer. Prompt length: {len(input_ids[0])} tokens. Max new: {config.LLM_MAX_NEW_TOKENS}.")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=config.LLM_MAX_NEW_TOKENS, temperature=float(config.LLM_TEMPERATURE)
                )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"LLM Answer (snippet): {answer[:150].replace(chr(10), ' ')}...")

            normalized_query = query.lower()
            current_llm_answer = answer.strip()

            # Password Reset Automation
            trigger_password_automation = False
            if any(keyword in normalized_query for keyword in PASSWORD_RESET_KEYWORDS):
                if len(current_llm_answer) < 75 and not any(k in current_llm_answer.lower() for k in PASSWORD_RESPONSE_KEYWORDS):
                    trigger_password_automation = True
                elif not any(k in current_llm_answer.lower() for k in PASSWORD_RESPONSE_KEYWORDS):
                    trigger_password_automation = True
            if trigger_password_automation:
                service_name = None
                if re.search(r"(?:my|the|a|an)\s+(windows)\s+password", normalized_query): service_name = "Windows"
                else:
                    m = re.search(r"password\s+(?:for|on|to|with)\s+([a-zA-Z0-9_.-]+)", normalized_query) or \
                        re.search(r"([a-zA-Z0-9_.-]+)\s+password", normalized_query)
                    if m:
                        ps = m.group(1)
                        if ps not in ["my", "the", "account", "email", "e-mail", "current"]: service_name = ps
                logger.info(f"Password automation triggered. Service: {service_name}")
                guidance = get_password_reset_guidance(service_name)
                current_llm_answer += cls._format_structured_guidance(guidance)

            # Connectivity Check Automation
            trigger_connectivity_automation = False
            if any(keyword in normalized_query for keyword in CONNECTIVITY_KEYWORDS):
                if len(current_llm_answer) < 100 and not any(k in current_llm_answer.lower() for k in CONNECTIVITY_RESPONSE_KEYWORDS):
                    trigger_connectivity_automation = True
                elif not any(k in current_llm_answer.lower() for k in CONNECTIVITY_RESPONSE_KEYWORDS):
                     trigger_connectivity_automation = True
            if trigger_connectivity_automation:
                logger.info("Connectivity automation triggered.")
                guidance = get_connectivity_check_guidance()
                current_llm_answer += cls._format_structured_guidance(guidance)

            return current_llm_answer.strip()
        except Exception as e:
            logger.error(f"Error in LLM generation or automation: {e}", exc_info=True)
            return "Error: LLM encountered issue generating answer or applying automation."

if __name__ == '__main__':
    import unittest
    from unittest.mock import patch, MagicMock

    # Basic logging for tests
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # --- Test _format_structured_guidance ---
    logger.info("\n--- Testing _format_structured_guidance ---")
    test_guidance_data_full = {
        "title": "Test Guide",
        "steps": ["Step 1 details", "Step 2 details"],
        "notes": ["Note A", "Note B"]
    }
    formatted_full = LLMGenerator._format_structured_guidance(test_guidance_data_full)
    assert "**Automated Guide: Test Guide**" in formatted_full
    assert "1. Step 1 details" in formatted_full
    assert "- Note A" in formatted_full
    logger.info("Test _format_structured_guidance (full) Passed.")

    test_guidance_data_no_notes = {"title": "NoNotes Guide", "steps": ["Step Alpha"]}
    formatted_no_notes = LLMGenerator._format_structured_guidance(test_guidance_data_no_notes)
    assert "**Important Notes:**" not in formatted_no_notes
    logger.info("Test _format_structured_guidance (no notes) Passed.")

    assert LLMGenerator._format_structured_guidance({}) == "", "Empty dict should yield empty string"
    assert LLMGenerator._format_structured_guidance({"title":"Only title"}) == "", "Dict with only title should be empty string"
    logger.info("Test _format_structured_guidance (empty/partial) Passed.")

    # --- Test format_prompt ---
    logger.info("\n--- Testing format_prompt ---")
    test_template = "CTX: {context_chunks} HIST: {conversation_history} Q: {question}"
    fp_res1 = LLMGenerator.format_prompt("Query1", ["Ctx1"], test_template, ["Hist1"])
    assert "CTX: Ctx1" in fp_res1 and "HIST: Turn 1: Hist1" in fp_res1 and "Q: Query1" in fp_res1
    logger.info("Test format_prompt (with all) Passed.")
    fp_res2 = LLMGenerator.format_prompt("Query2", [], test_template, [])
    assert "CTX: No specific context" in fp_res2 and "HIST: No prior conversation" in fp_res2
    logger.info("Test format_prompt (no context/history) Passed.")

    # --- Test _initialize and get_llm_resources (with mocks) ---
    logger.info("\n--- Testing _initialize & get_llm_resources ---")
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_load, \
         patch('transformers.AutoModelForSeq2SeqLM.from_pretrained') as mock_model_load:

        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.device = "mock_device" # Mock the device attribute

        mock_tokenizer_load.return_value = mock_tokenizer_instance
        mock_model_load.return_value = mock_model_instance

        # Reset class state for clean test
        LLMGenerator._model = None
        LLMGenerator._tokenizer = None
        LLMGenerator._model_name_loaded = None

        test_model_name = "test/model"
        config.LLM_MODEL_NAME = test_model_name # Override config for this test

        LLMGenerator.get_llm_resources() # First call initializes
        mock_tokenizer_load.assert_called_once_with(test_model_name)
        mock_model_load.assert_called_once_with(test_model_name, device_map=LLMGenerator._device) # device is determined internally
        assert LLMGenerator._model == mock_model_instance
        assert LLMGenerator._tokenizer == mock_tokenizer_instance
        assert LLMGenerator._model_name_loaded == test_model_name
        logger.info("Test _initialize (first call) Passed.")

        LLMGenerator.get_llm_resources() # Second call should use cached
        mock_tokenizer_load.assert_called_once() # Not called again
        mock_model_load.assert_called_once()   # Not called again
        logger.info("Test get_llm_resources (cached call) Passed.")

    # --- Test generate_answer (with mocks for LLM and automations) ---
    logger.info("\n--- Testing generate_answer (with automations) ---")

    # Mock LLM resources
    mock_llm_tokenizer = MagicMock()
    mock_llm_model = MagicMock()
    mock_llm_model.device = "cpu" # Set a device for the mock model

    # Mock tokenizer call behavior
    mock_input_tensors = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    mock_llm_tokenizer.return_value = mock_input_tensors # Mock __call__
    mock_llm_tokenizer.model_max_length = 512

    # Patch get_llm_resources to return our mocks
    @patch.object(LLMGenerator, 'get_llm_resources', return_value=(mock_llm_model, mock_llm_tokenizer))
    def run_generate_answer_tests(mock_get_resources):

        # Test Case: Password reset query, LLM gives generic response
        mock_llm_model.generate.return_value = torch.tensor([[10, 20, 30]]) # Dummy output tokens
        mock_llm_tokenizer.decode.return_value = "I am here to help you." # Generic response

        query_pwd = "I forgot my password for my_service"
        answer_pwd = LLMGenerator.generate_answer(query_pwd, [], [])
        assert "Automated Guide: Password Reset Guide for my_service" in answer_pwd
        assert "I am here to help you." in answer_pwd # Appended
        logger.info("Test generate_answer (password reset - generic LLM) Passed.")

        # Test Case: Password reset query, LLM gives good response
        mock_llm_tokenizer.decode.return_value = "Sure, I can help you reset your password. What service is it for?"
        answer_pwd_good_llm = LLMGenerator.generate_answer(query_pwd, [], [])
        assert "Automated Guide: Password Reset Guide" not in answer_pwd_good_llm # Should NOT append
        assert "Sure, I can help you reset your password" in answer_pwd_good_llm
        logger.info("Test generate_answer (password reset - good LLM) Passed.")

        # Test Case: Connectivity query, LLM gives generic response
        mock_llm_tokenizer.decode.return_value = "Tell me more."
        query_conn = "My internet is not working"
        answer_conn = LLMGenerator.generate_answer(query_conn, [], [])
        assert "Automated Guide: Basic Connectivity Troubleshooting Guide" in answer_conn
        assert "Tell me more." in answer_conn
        logger.info("Test generate_answer (connectivity - generic LLM) Passed.")

        # Test Case: Connectivity query, LLM gives good response
        mock_llm_tokenizer.decode.return_value = "Okay, let's check your router and modem."
        answer_conn_good_llm = LLMGenerator.generate_answer(query_conn, [], [])
        assert "Automated Guide: Basic Connectivity Troubleshooting Guide" not in answer_conn_good_llm
        assert "Okay, let's check your router and modem." in answer_conn_good_llm
        logger.info("Test generate_answer (connectivity - good LLM) Passed.")

        # Test Case: No specific automation keywords
        mock_llm_tokenizer.decode.return_value = "This is a standard answer."
        query_normal = "What is a firewall?"
        answer_normal = LLMGenerator.generate_answer(query_normal, [], [])
        assert "Automated Guide" not in answer_normal
        assert "This is a standard answer." in answer_normal
        logger.info("Test generate_answer (no automation) Passed.")

    run_generate_answer_tests()

    logger.info("\n--- LLMHandler Test Suite Complete ---")
```
