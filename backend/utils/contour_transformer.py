from typing import Tuple
import numpy as np
from pickle import load


class ContourTransformationModels:
    def __init__(self, model_file):
        self.model_tx, self.model_ty, self.model_bias, self.poly_transformer = load(
            model_file
        )

    def predict_transformations(
        self, scene_depth: float, object_depth: float
    ) -> Tuple[float, float, float]:
        """
        Predict tx, ty, bias from given scene depth and object depth

        Args:
            scene_depth (float): Depth of the scene.
            object_depth (float): Depth of the object.

        Returns:
            Tuple[float, float, float]: tx, ty, bias
        """
        input_x = np.array([[scene_depth, object_depth]])

        input_poly = self.poly_transformer.transform(input_x)

        return (
            self.model_tx.predict(input_poly)[0],
            self.model_ty.predict(input_poly)[0],
            self.model_bias.predict(input_poly)[0],
        )
