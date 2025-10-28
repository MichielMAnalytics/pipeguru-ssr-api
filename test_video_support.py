"""Quick test script to verify video support implementation."""

import base64

# Test MIME detection
from src.utils.mime_detector import detect_mime_type_from_base64, validate_mime_type

# Create sample base64 for different formats
test_cases = [
    # JPEG image
    {
        "name": "JPEG image",
        "data": base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 100).decode(),
        "expected_mime": "image/jpeg",
        "expected_category": "image",
    },
    # PNG image
    {
        "name": "PNG image",
        "data": base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode(),
        "expected_mime": "image/png",
        "expected_category": "image",
    },
    # MP4 video
    {
        "name": "MP4 video",
        "data": base64.b64encode(b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 100).decode(),
        "expected_mime": "video/mp4",
        "expected_category": "video",
    },
    # WebM video
    {
        "name": "WebM video",
        "data": base64.b64encode(b"\x1aE\xdf\xa3" + b"\x00" * 100).decode(),
        "expected_mime": "video/webm",
        "expected_category": "video",
    },
]

print("Testing MIME type detection...\n")

passed = 0
failed = 0

for test in test_cases:
    try:
        mime_type, category = detect_mime_type_from_base64(test["data"])

        if mime_type == test["expected_mime"] and category == test["expected_category"]:
            print(f"✅ {test['name']}: PASSED")
            print(f"   Detected: {mime_type} ({category})")
            passed += 1
        else:
            print(f"❌ {test['name']}: FAILED")
            print(f"   Expected: {test['expected_mime']} ({test['expected_category']})")
            print(f"   Got: {mime_type} ({category})")
            failed += 1
    except Exception as e:
        print(f"❌ {test['name']}: ERROR - {str(e)}")
        failed += 1

    print()

print(f"\nResults: {passed} passed, {failed} failed")

# Test MIME validation
print("\n" + "="*50)
print("Testing MIME type validation...\n")

validation_tests = [
    ("video/mp4", True),
    ("image/jpeg", True),
    ("video/quicktime", True),
    ("image/png", True),
    ("video/avi", True),
    ("application/pdf", False),  # Unsupported
    ("text/plain", False),  # Unsupported
]

for mime, should_pass in validation_tests:
    try:
        normalized, category = validate_mime_type(mime)
        if should_pass:
            print(f"✅ {mime}: Valid → {normalized} ({category})")
        else:
            print(f"❌ {mime}: Should have failed but passed")
    except ValueError as e:
        if not should_pass:
            print(f"✅ {mime}: Correctly rejected")
        else:
            print(f"❌ {mime}: Should have passed but failed - {str(e)}")

print("\n✅ All MIME detection tests completed!")
