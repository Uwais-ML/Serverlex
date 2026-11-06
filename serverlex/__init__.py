from serverlex.preprocessing.clean import Clean
from serverlex.preprocessing.pipeline import MLPipeline
from serverlex.diagnostics.dataguard import DataGuard
from serverlex.diagnostics.evaluation import ModelEvaluator
from serverlex.diagnostics.explainability import ModelExplainer
from serverlex.diagnostics.drift import DriftDetector
from serverlex.modeling.hyperparameter import ModelHyperparameter
from serverlex.modeling.ensemble import EnsembleBuilder
from serverlex.modeling.selector import FeatureSelector
from serverlex.visualization.eda import Visualize
from serverlex.serving.deployer import ModelDeployer
from serverlex.copilot.agent import ServerlexCopilot, get_copilot
from serverlex.copilot.hook import enable_auto_copilot, disable_auto_copilot

__all__ = [
    "Clean",
    "MLPipeline",
    "DataGuard",
    "ModelEvaluator",
    "ModelExplainer",
    "DriftDetector",
    "ModelHyperparameter",
    "EnsembleBuilder",
    "FeatureSelector",
    "Visualize",
    "ModelDeployer",
    "ServerlexCopilot",
    "get_copilot",
    "enable_auto_copilot",
    "disable_auto_copilot",
]

__version__ = "1.3.3"
