from typing import Dict, Optional, List, Any # Added Any

# Added logging import for the __main__ block, though not strictly necessary for the functions themselves
import logging

def get_password_reset_guidance(service_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Provides structured guidance for password resets, with some service-specific examples.

    Args:
        service_name: Optional name of the service for which password reset is requested.

    Returns:
        A dictionary containing 'title', 'steps' (a list of strings), and 'notes' (a list of strings).
    """
    title = "Password Reset Guide"
    steps: List[str] = []
    notes: List[str] = [
        "Ensure you are on a secure computer and internet connection.",
        "Do not share your password with anyone.",
        "Choose a strong password that you haven't used before.",
        "Consider using a password manager to help you remember complex passwords."
    ]

    normalized_service_name = service_name.lower() if service_name else ""

    if "windows" in normalized_service_name:
        title = "Windows Password Reset Guide"
        steps = [
            "If you're at the login screen: Look for a 'Reset password' or 'I forgot my password' link. This usually requires a password reset disk (if created earlier) or answering security questions if you're using a Microsoft account.",
            "If using a Microsoft account to log in: You can reset your Microsoft account password online. Go to account.live.com/password/reset and follow the instructions.",
            "If using a local account and you don't have a reset disk or can't answer security questions: You might need administrator assistance or specialized tools. Please contact your IT administrator if this is a work computer.",
            "For domain-joined computers: Usually, you'll need to contact your IT department to reset your domain password."
        ]
        notes.append("Windows password reset procedures can vary significantly based on how your account is set up (local, Microsoft account, domain account).")
    elif "google" in normalized_service_name or "gmail" in normalized_service_name:
        title = "Google/Gmail Account Password Reset Guide"
        steps = [
            "Go to the Google account recovery page: g.co/recover.",
            "Enter your email address or phone number associated with your account.",
            "Follow the on-screen instructions. Google will ask you some questions to confirm it's your account or send a verification code to your recovery email or phone.",
            "Once verified, you'll be prompted to create a new password."
        ]
    elif "microsoft account" in normalized_service_name: # Different from local Windows login
        title = "Microsoft Account Password Reset Guide"
        steps = [
            "Go to the Microsoft account password reset page: account.live.com/password/reset.",
            "Enter your email, phone, or Skype name associated with your Microsoft account.",
            "Follow the instructions. Microsoft will guide you through verifying your identity, often by sending a code to a recovery email or phone number.",
            "After verification, you can create a new password."
        ]
    elif "web service" in normalized_service_name or "website" in normalized_service_name or not service_name: # Generic or if service_name is "generic web service"
        if service_name and service_name.lower() not in ["web service", "website"]:
             title = f"Password Reset Guide for {service_name}"
        else:
            title = "Generic Web Service Password Reset Guide"
        steps = [
            "Go to the login page of the website or service.",
            "Look for a link that says 'Forgot password?', 'Reset password', 'Can't log in?', or similar. This is usually near the password field.",
            "Click the link and follow the on-screen instructions. You'll typically be asked to enter your email address or username.",
            "Check your email (including spam/junk folder) for a password reset link or code from the service.",
            "Click the link or enter the code and follow the prompts to create a new password."
        ]
        if not service_name: # Add this note only if it's truly generic
             notes.insert(0, "These are general steps. The exact process may vary slightly from one website/service to another.")

    else: # Service name provided but no specific guide
        title = f"Password Reset Guide for {service_name}"
        steps = [
            f"For '{service_name}', typically you would go to their main login page.",
            "Look for a link such as 'Forgot password?', 'Reset password', or 'Account recovery'.",
            "Follow the instructions provided by the service. This often involves verifying your identity via email or a registered phone number.",
            "If you cannot find this option, try searching for '{service_name} password reset' on a search engine."
        ]
        notes.insert(0, f"The exact steps for '{service_name}' might differ. Check their official help pages if available.")

    return {"title": title, "steps": steps, "notes": notes}


def get_connectivity_check_guidance() -> Dict[str, Any]:
    """
    Provides structured guidance for basic network connectivity troubleshooting.

    Returns:
        A dictionary containing 'title', 'steps' (a list of strings), and 'notes' (a list of strings).
    """
    title = "Basic Connectivity Troubleshooting Guide"
    steps: List[str] = [
        "**Check Physical Connections:** Ensure all network cables (Ethernet, modem, router) are securely plugged in. If using Wi-Fi, make sure you're connected to the correct network and have entered the password correctly.",
        "**Restart Your Modem and Router:** Unplug your modem and router for about 30 seconds. Plug in the modem first, wait for it to fully power on (usually 1-2 minutes, all lights stable), then plug in your router and wait for it to power on.",
        "**Restart Your Computer/Device:** A simple reboot can often resolve temporary network glitches.",
        "**Check Network Status Icon:** Look at the network icon (Wi-Fi bars or Ethernet symbol) on your computer. Does it show any errors (e.g., a yellow exclamation mark, a red X, or 'No Internet')? Hover over it or click it for more details.",
        "**Run Network Troubleshooter (Windows/Mac):** Most operating systems have a built-in network troubleshooter. Right-click the network icon and look for an option like 'Troubleshoot problems' (Windows) or use 'Wireless Diagnostics' (Mac).",
        "**Test with a Ping Command:** Open Command Prompt (Windows) or Terminal (Mac/Linux). Type `ping 8.8.8.8` and press Enter. If you see replies, your basic internet connectivity is likely working. If you see 'Request timed out' or errors, there's a connectivity problem.",
        "  - You can also try pinging a website like `ping google.com` to check DNS resolution as well.",
        "**Check IP Configuration:** Open Command Prompt (Windows) and type `ipconfig /all` or Terminal (Mac/Linux) and type `ifconfig` (or `ip addr`). Look for an IPv4 address (e.g., 192.168.1.x or 10.0.0.x). If it starts with 169.254.x.x, your device isn't getting a valid IP address from the router.",
        "**Try Another Device:** If possible, check if other devices (phone, another computer) on the same network can access the internet. This helps determine if the issue is with your specific device or the broader network/internet service."
    ]
    notes: List[str] = [
        "If you're using a VPN, try disconnecting from it temporarily to see if that resolves the issue.",
        "Firewalls or security software can sometimes block internet access. Temporarily disabling them for testing (if safe to do so) might help identify the cause.",
        "If only one website or service is inaccessible, the problem might be with that specific site, not your internet connection.",
        "If problems persist after these steps, contact your Internet Service Provider (ISP) or IT department for further assistance."
    ]
    return {"title": title, "steps": steps, "notes": notes}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Changed print_guidance to take the dict directly for more flexibility
    def print_guidance_from_dict(guidance_dict: Dict[str, Any]):
        logger.info(f"\n--- {guidance_dict['title']} ---")
        logger.info("Steps:")
        for i, step in enumerate(guidance_dict['steps']):
            logger.info(f"  {i+1}. {step}")
        if guidance_dict['notes']:
            logger.info("Notes:")
            for note in guidance_dict['notes']:
                logger.info(f"  - {note}")

    print_guidance_from_dict(get_password_reset_guidance(None)) # Generic
    print_guidance_from_dict(get_password_reset_guidance("Windows"))
    print_guidance_from_dict(get_password_reset_guidance("Google"))
    print_guidance_from_dict(get_password_reset_guidance("MyCustomApp"))
    print_guidance_from_dict(get_password_reset_guidance("Generic Web Service"))

    # Test connectivity guidance
    print_guidance_from_dict(get_connectivity_check_guidance())
```
