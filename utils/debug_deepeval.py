#!/usr/bin/env python3
"""
Debug script for DeepEval connection issues
This helps identify exactly what's going wrong with the DeepEval API
"""

import os
import sys

# Set API key
DEEP_EVAL_KEY = "confident_us_mFcVdsaTXyg+6AlOkiu8QdCOrNs1vpnc+MhpgWpObzo="
os.environ["DEEPEVAL_API_KEY"] = DEEP_EVAL_KEY

def test_basic_connection():
    """Test basic DeepEval connection"""
    print("🔌 Testing basic DeepEval connection...")
    
    try:
        from deepeval.prompt import Prompt
        print("✅ DeepEval import successful")
        
        # Check API key
        api_key = os.getenv("DEEPEVAL_API_KEY")
        if api_key:
            print(f"✅ API key set (length: {len(api_key)})")
            if api_key.startswith("confident_"):
                print("✅ API key format looks correct")
            else:
                print("⚠️ API key format might be incorrect")
        else:
            print("❌ API key not found")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import DeepEval: {e}")
        print("💡 Try: pip install deepeval")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_prompt_creation():
    """Test creating a Prompt instance"""
    print("\n🏗️ Testing Prompt instance creation...")
    
    try:
        from deepeval.prompt import Prompt
        
        # Test creating prompt instance
        prompt_instance = Prompt(alias="data_fetcher")
        print("✅ Prompt instance created successfully")
        print(f"📝 Prompt alias: {prompt_instance.alias}")
        
        # Check available methods
        methods = [method for method in dir(prompt_instance) if not method.startswith('_')]
        print(f"📋 Available methods: {', '.join(methods)}")
        
        # Check for pull method specifically
        if hasattr(prompt_instance, 'pull'):
            print("✅ 'pull' method available")
        else:
            print("❌ 'pull' method not found")
            
        if hasattr(prompt_instance, 'push'):
            print("✅ 'push' method available")
        else:
            print("⚠️ 'push' method not available (this is expected in newer versions)")
            
        return prompt_instance
        
    except Exception as e:
        print(f"❌ Failed to create Prompt instance: {e}")
        return None

def test_prompt_pull_variations(prompt_instance):
    """Test different ways to pull the prompt"""
    print("\n📥 Testing different pull methods...")
    
    if not prompt_instance:
        print("❌ No prompt instance to test with")
        return
    
    test_cases = [
        ("Basic pull()", lambda p: p.pull()),
        ("Pull with v1", lambda p: p.pull(version="v1")),
        ("Pull with fallback=True", lambda p: p.pull(version="v1", fallback_to_cache=True)),
        ("Pull with fallback=False", lambda p: p.pull(version="v1", fallback_to_cache=False)),
    ]
    
    for test_name, test_func in test_cases:
        try:
            print(f"\n🔄 Testing: {test_name}")
            test_func(prompt_instance)
            
            # Check if we got text
            if hasattr(prompt_instance, 'text') and prompt_instance.text:
                print(f"✅ Success! Got {len(prompt_instance.text)} characters")
                print(f"📝 Preview: {prompt_instance.text[:100]}...")
                return True  # Success, no need to try other methods
            else:
                print("⚠️ Pull succeeded but no text found")
                
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            # Print more details about the error
            print(f"🔍 Error type: {type(e).__name__}")
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"🔍 Root cause: {e.__cause__}")
    
    return False

def test_cache_directory():
    """Check DeepEval cache directory"""
    print("\n📁 Checking DeepEval cache...")
    
    # Common cache locations
    possible_cache_paths = [
        os.path.expanduser("~/.deepeval"),
        os.path.expanduser("~/.cache/deepeval"),
        ".deepeval",
        "deepeval_cache"
    ]
    
    for cache_path in possible_cache_paths:
        if os.path.exists(cache_path):
            print(f"✅ Found cache directory: {cache_path}")
            try:
                files = os.listdir(cache_path)
                print(f"📋 Cache contents: {files}")
                
                # Look for prompt files
                prompt_files = [f for f in files if 'prompt' in f.lower() or 'data_fetcher' in f.lower()]
                if prompt_files:
                    print(f"🎯 Found prompt-related files: {prompt_files}")
                else:
                    print("⚠️ No prompt files found in cache")
                    
            except Exception as e:
                print(f"❌ Error reading cache directory: {e}")
        else:
            print(f"❌ Cache directory not found: {cache_path}")

def test_api_connectivity():
    """Test if we can reach the DeepEval API"""
    print("\n🌐 Testing API connectivity...")
    
    try:
        import requests
        
        # Test basic connectivity (this is a guess at the API endpoint)
        api_base = "https://api.confident-ai.com"  # or whatever the actual endpoint is
        
        print(f"🔍 Testing connection to {api_base}")
        
        # Just test basic connectivity, don't make actual API calls
        response = requests.get(api_base, timeout=5)
        print(f"✅ Can reach API endpoint (status: {response.status_code})")
        
    except ImportError:
        print("⚠️ 'requests' package not available for API testing")
    except Exception as e:
        print(f"❌ API connectivity test failed: {e}")

def main():
    """Run all debug tests"""
    print("🚀 DeepEval Connection Debug Tool")
    print("=" * 50)
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("\n❌ Basic connection failed. Stopping tests.")
        return
    
    # Test 2: Prompt creation
    prompt_instance = test_prompt_creation()
    
    # Test 3: Pull variations
    success = test_prompt_pull_variations(prompt_instance)
    
    # Test 4: Cache directory
    test_cache_directory()
    
    # Test 5: API connectivity
    test_api_connectivity()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 At least one pull method worked!")
        print("💡 Use the successful method in your application")
    else:
        print("❌ All pull methods failed")
        print("💡 Recommendations:")
        print("  1. Check your internet connection")
        print("  2. Verify your API key is correct")
        print("  3. Make sure the 'data_fetcher' prompt exists in your DeepEval project")
        print("  4. Try creating the prompt via the web interface first")
        print("  5. Use the fallback prompt in your application")

if __name__ == "__main__":
    main()