from PIL import Image
from anode import image_only
from seed import run

def test_image_only_direct():
    """Test the image_only function directly"""
    try:
        # Load a test image
        image = Image.open("test_image.jpg")
        
        # Test the image_only function
        result = image_only(image)
        print("Direct image_only result:")
        print(result)
        print("\n" + "-"*50 + "\n")
        
        return True
    except Exception as e:
        print(f"Error in test_image_only_direct: {e}")
        return False

def test_run_with_question():
    """Test the run function with a question"""
    try:
        # Load a test image
        image = Image.open("test_image.jpg")
        
        # Test with a question
        result = run(image, "What objects do you see in this image?")
        print("run with question result:")
        print(result)
        print("\n" + "-"*50 + "\n")
        
        return True
    except Exception as e:
        print(f"Error in test_run_with_question: {e}")
        return False

def test_run_without_question():
    """Test the run function without a question (image-only mode)"""
    try:
        # Load a test image
        image = Image.open("test_image.jpg")
        
        # Test without a question
        result = run(image)
        print("run without question result:")
        print(result)
        print("\n" + "-"*50 + "\n")
        
        return True
    except Exception as e:
        print(f"Error in test_run_without_question: {e}")
        return False

if __name__ == "__main__":
    print("Testing image-only model integration...")
    
    # Run tests
    direct_test = test_image_only_direct()
    question_test = test_run_with_question()
    no_question_test = test_run_without_question()
    
    # Print summary
    print("Test Results:")
    print(f"Direct image_only test: {'PASSED' if direct_test else 'FAILED'}")
    print(f"Run with question test: {'PASSED' if question_test else 'FAILED'}")
    print(f"Run without question test: {'PASSED' if no_question_test else 'FAILED'}")
