from scripts.train import main
from src.datasets.speech_enhancement import SpeechEnhancementDataset, generate_and_save_noisy_audio, prepare_dataset_structure
from src.losses.speech_enhancement_loss import compute_loss
from src.models.dual_branch_transformer import DualBranchTransformer
from src.trainers.workflow import build_dataloaders, build_datasets, train_model, train_one_epoch, validate
from src.utils.train_eval import evaluate_speech_quality, generate_enhancement_report, save_enhanced_audio, visualize_spectrogram_comparison


if __name__ == "__main__":
    main()
