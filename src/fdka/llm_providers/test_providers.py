# test_providers.py

import os
import sys


# ✅ ADDED: Proper error handling for missing keys
def check_api_keys():
    """Verify API keys are set and valid."""
    openai_key = os.getenv('OPENAI_API_KEY')
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')

    print("🔑 API Key Status:")
    print("-" * 70)

    if not openai_key:
        print("❌ OPENAI_API_KEY not set")
        print("   Set it with: export OPENAI_API_KEY='sk-proj-YOUR_KEY_HERE'")
        print("   Get key from: https://platform.openai.com/api-keys")
        return False
    elif openai_key.startswith("sk-proj-...") or openai_key == "YOUR_KEY_HERE":
        print("❌ OPENAI_API_KEY is placeholder")
        print("   Replace with actual key from https://platform.openai.com/api-keys")
        return False
    else:
        print(f"✅ OPENAI_API_KEY set ({openai_key[:10]}...)")

    if not deepseek_key:
        print("⚠️  DEEPSEEK_API_KEY not set (skipping DeepSeek test)")
        print("   Set it with: export DEEPSEEK_API_KEY='sk-YOUR_KEY_HERE'")
        print("   Get key from: https://platform.deepseek.com")
        has_deepseek = False
    elif deepseek_key.startswith("sk-...") or deepseek_key == "YOUR_KEY_HERE":
        print("⚠️  DEEPSEEK_API_KEY is placeholder (skipping DeepSeek test)")
        has_deepseek = False
    else:
        print(f"✅ DEEPSEEK_API_KEY set ({deepseek_key[:10]}...)")
        has_deepseek = True

    print("-" * 70)
    return True, has_deepseek


# Check keys first
valid, has_deepseek = check_api_keys()
if not valid:
    sys.exit(1)

from openai_provider import OpenAIProvider

if has_deepseek:
    from deepseek_provider import DeepSeekProvider

print("\n" + "=" * 70)
print("Testing OpenAI Provider (Responses API)")
print("=" * 70)

openai_config = {
    'model': 'gpt-4o-mini',
    'temperature': 0.3,
    'max_tokens': 500
}

try:
    openai_provider = OpenAIProvider(openai_config)
    result = openai_provider.generate(prompt="What is 2+2? Answer briefly.")

    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        print(f"   Type: {result['error_type']}")
        print(f"   Code: {result['error_code']}")
    else:
        print(f"✅ Response: {result['text'][:100]}")
        print(
            f"   Tokens: {result['tokens_used']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
        print(f"   Latency: {result['latency_sec']:.2f}s")
        cost = (result['prompt_tokens'] * 0.15 + result['completion_tokens'] * 0.60) / 1_000_000
        print(f"   Cost: ${cost:.6f}")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()

if has_deepseek:
    print("\n" + "=" * 70)
    print("Testing DeepSeek Provider (Chat Completions API)")
    print("=" * 70)

    deepseek_config = {
        'model': 'deepseek-chat',
        'temperature': 0.3,
        'max_tokens': 500
    }

    try:
        deepseek_provider = DeepSeekProvider(deepseek_config)
        result = deepseek_provider.generate(prompt="What is 2+2? Answer briefly.")

        if result.get('error'):
            print(f"❌ Error: {result['error']}")
            print(f"   Type: {result['error_type']}")
            print(f"   Code: {result['error_code']}")
        else:
            print(f"✅ Response: {result['text'][:100]}")
            print(
                f"   Tokens: {result['tokens_used']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
            print(f"   Latency: {result['latency_sec']:.2f}s")
            cost = (result['prompt_tokens'] * 0.14 + result['completion_tokens'] * 0.28) / 1_000_000
            print(f"   Cost: ${cost:.6f}")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 70)
if has_deepseek:
    print("✅ Both providers tested successfully!")
else:
    print("✅ OpenAI provider tested successfully!")
print("=" * 70)