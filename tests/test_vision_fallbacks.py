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

    def test_prepare_batch_uploads_keeps_all_images_for_large_zips(self):
        module = importlib.import_module("pages.Vision_AI")
        batch_uploads = [{"filename": f"img{i}.jpg"} for i in range(12)]

        prepared_uploads = module._prepare_batch_uploads(batch_uploads)

        self.assertEqual(len(prepared_uploads), 12)
        self.assertEqual(prepared_uploads[0]["filename"], "img0.jpg")
        self.assertEqual(prepared_uploads[-1]["filename"], "img11.jpg")



if __name__ == "__main__":
    unittest.main()
