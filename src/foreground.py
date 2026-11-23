from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy import ndimage


class ForegroundSegmenter:
    def __init__(self, model: Path, detail_model: Path):
        assert model.is_file() and detail_model.is_file(), (model, detail_model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        assert self.session.get_inputs()[0].shape == [1, 3, 320, 320]
        self.detail_session = ort.InferenceSession(str(detail_model), sess_options=options,
                                                   providers=['CPUExecutionProvider'])
        self.detail_input_name = self.detail_session.get_inputs()[0].name
        assert self.detail_session.get_inputs()[0].shape == [1, 3, 1024, 1024]
        self.model = {'u2net': str(model.resolve()), 'isnet': str(detail_model.resolve())}

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 [H,W,3] -> foreground probability float32 [H,W], using U2Net."""
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        resized = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        normalized = resized / max(float(resized.max()), 1e-6)
        normalized = (normalized - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        prediction = self.session.run(None, {self.input_name: tensor})[0][0, 0]
        assert np.isfinite(prediction).all() and np.ptp(prediction) > 1e-6
        prediction = (prediction - prediction.min()) / np.ptp(prediction)
        return cv2.resize(prediction, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR).clip(0, 1)

    def predict_detail(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 [1024,1024,3] -> foreground probability float32 [1024,1024], using IS-Net."""
        assert rgb.dtype == np.uint8 and rgb.shape == (1024, 1024, 3)
        normalized = rgb.astype(np.float32) / max(float(rgb.max()), 1e-6) - .5
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        prediction = self.detail_session.run(None, {self.detail_input_name: tensor})[0][0, 0]
        assert np.isfinite(prediction).all() and np.ptp(prediction) > 1e-6
        return ((prediction - prediction.min()) / np.ptp(prediction)).clip(0, 1)

    def multiview(self, views: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Four RGB uint8 [1024,1024,3] views -> float32 union alpha, bool masks [4,H,W], float32 model alpha [2,4,H,W]."""
        assert len(views) == 4 and all(view.shape == (1024, 1024, 3) for view in views)
        model_alpha = np.stack(([self.predict(view) for view in views],
                                [self.predict_detail(view) for view in views]))
        alpha = model_alpha.max(axis=0)
        masks = np.stack([ndimage.binary_dilation(a >= .5, iterations=3) for a in alpha])
        assert masks.reshape(4, -1).any(axis=1).all(), 'Foreground segmentation returned an empty view'
        return alpha, masks, model_alpha
