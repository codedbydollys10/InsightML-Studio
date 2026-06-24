import importlib
import sys
import unittest


class VisionImportTests(unittest.TestCase):
    def test_importing_page_does_not_load_vision_runtime(self):
        for name in ["pages.Vision_AI", "utils.vision", "utils.object_detection", "utils.scene_analysis"]:
            sys.modules.pop(name, None)

        module = importlib.import_module("pages.Vision_AI")

        self.assertIn("pages.Vision_AI", sys.modules)
        self.assertNotIn("utils.vision", sys.modules)
        self.assertNotIn("utils.object_detection", sys.modules)
        self.assertNotIn("utils.scene_analysis", sys.modules)
        self.assertTrue(hasattr(module, "render"))


if __name__ == "__main__":
    unittest.main()
