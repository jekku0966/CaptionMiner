"""A deliberately focused drag-and-drop desktop interface."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from captionminer import __version__
from captionminer.config import COMMON_LANGUAGES, MODEL_PROFILES, options_for_profile
from captionminer.model_management import (
    DownloadConsentAction,
    DownloadPolicy,
    ModelPreferences,
    ModelSelection,
    apply_download_consent_action,
    huggingface_cache_directory,
    resolve_installed_model,
)
from captionminer.pipeline import transcribe_to_srt
from captionminer.progress import INDETERMINATE_PROGRESS, batch_progress_value, format_elapsed
from captionminer.transcribe import TranscriptionCancelled, TranscriptionEngine


class MediaList(QListWidget):
    paths_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setMinimumHeight(170)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.paths_dropped.emit(paths)
        event.acceptProposedAction()


class BatchWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    completed = Signal(int, int, bool)

    def __init__(
        self,
        files: list[Path],
        *,
        profile: str,
        language: str | None,
        device: str,
        initial_prompt: str | None,
        output_directory: Path | None,
        overwrite: bool,
        model_reference: str,
        local_files_only: bool,
    ) -> None:
        super().__init__()
        self.files = files
        base_options = options_for_profile(
            profile,
            language=language,
            device=device,
            initial_prompt=initial_prompt,
        )
        self.options = replace(
            base_options,
            model_name=model_reference,
            local_files_only=local_files_only,
        )
        self.output_directory = output_directory
        self.overwrite = overwrite
        self.cancel_event = threading.Event()

    @Slot()
    def run(self) -> None:
        completed = 0
        failed = 0
        engine = TranscriptionEngine(self.options)
        total = len(self.files)

        for file_index, source in enumerate(self.files):
            if self.cancel_event.is_set():
                break
            self.log.emit(f"[{file_index + 1}/{total}] {source}")

            def callback(
                fraction: float | None,
                message: str,
                current_index: int = file_index,
            ) -> None:
                if fraction is None:
                    self.log.emit(message)
                overall = batch_progress_value(current_index, total, fraction)
                self.progress.emit(overall, f"[{current_index + 1}/{total}] {message}")

            try:
                result = transcribe_to_srt(
                    engine,
                    source,
                    output_directory=self.output_directory,
                    overwrite=self.overwrite,
                    progress=callback,
                    cancel=self.cancel_event.is_set,
                )
                completed += 1
                language = result.metadata.language or "unknown language"
                recovery = (
                    f", recovered {result.metadata.recovered_word_count} word(s)"
                    if result.metadata.recovered_word_count
                    else ""
                )
                self.log.emit(
                    f"Created {result.output} ({result.cue_count} cues, {language}, "
                    f"{result.metadata.device}{recovery})."
                )
            except TranscriptionCancelled:
                self.log.emit("Cancellation requested; no partial SRT was written.")
                break
            except Exception as exc:
                failed += 1
                self.log.emit(f"ERROR: {source.name}: {exc}")

        cancelled = self.cancel_event.is_set()
        processed = completed + failed
        self.progress.emit(
            100 if not cancelled else round(processed / total * 100),
            "Finished" if not cancelled else "Cancelled",
        )
        self.completed.emit(completed, failed, cancelled)

    @Slot()
    def cancel(self) -> None:
        self.cancel_event.set()


class ModelSettingsDialog(QDialog):
    """Small user-facing model and download-preference editor."""

    def __init__(
        self,
        preferences: ModelPreferences,
        parent: QWidget | None = None,
        *,
        initial_profile_key: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.preferences = preferences
        self.setWindowTitle("CaptionMiner Settings")
        self.resize(620, 330)

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Choose what CaptionMiner should do when a selected speech-recognition "
            "model is not already on this computer."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form = QFormLayout()
        self.policy_combo = QComboBox()
        self.policy_combo.addItem("Ask before downloading", DownloadPolicy.ASK)
        self.policy_combo.addItem("Download automatically", DownloadPolicy.ALLOW)
        self.policy_combo.addItem("Never download automatically", DownloadPolicy.DENY)
        policy_index = self.policy_combo.findData(self.preferences.download_policy)
        self.policy_combo.setCurrentIndex(max(0, policy_index))
        form.addRow("When a model is missing", self.policy_combo)

        self.profile_combo = QComboBox()
        for profile in MODEL_PROFILES.values():
            self.profile_combo.addItem(f"{profile.label} — {profile.model_name}", profile.key)
        if initial_profile_key is not None:
            profile_index = self.profile_combo.findData(initial_profile_key)
            if profile_index >= 0:
                self.profile_combo.setCurrentIndex(profile_index)
        form.addRow("Model profile", self.profile_combo)
        layout.addLayout(form)

        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        model_buttons = QHBoxLayout()
        self.choose_local_button = QPushButton("Choose local model folder")
        self.clear_local_button = QPushButton("Remove local selection")
        self.open_cache_button = QPushButton("Open downloaded model folder")
        model_buttons.addWidget(self.choose_local_button)
        model_buttons.addWidget(self.clear_local_button)
        model_buttons.addStretch(1)
        model_buttons.addWidget(self.open_cache_button)
        layout.addLayout(model_buttons)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.policy_combo.currentIndexChanged.connect(self._save_policy)
        self.profile_combo.currentIndexChanged.connect(self._refresh_status)
        self.choose_local_button.clicked.connect(self._choose_local_model)
        self.clear_local_button.clicked.connect(self._clear_local_model)
        self.open_cache_button.clicked.connect(self._open_cache)
        self._refresh_status()

    def _profile_key(self) -> str:
        return str(self.profile_combo.currentData())

    @Slot()
    def _save_policy(self) -> None:
        self.preferences.set_download_policy(self.policy_combo.currentData())

    @Slot()
    def _refresh_status(self) -> None:
        profile_key = self._profile_key()
        profile = MODEL_PROFILES[profile_key]
        lookup = resolve_installed_model(profile_key, profile.model_name, self.preferences)
        saved_path = self.preferences.local_model_path(profile_key)

        if lookup.selection is not None and lookup.selection.source == "local":
            text = f"Using local model folder:\n{lookup.selection.location}"
        elif lookup.selection is not None:
            text = f"Downloaded model found:\n{lookup.selection.location}"
        elif lookup.invalid_local_reason:
            text = f"Saved local model cannot be used:\n{lookup.invalid_local_reason}"
        else:
            text = "This model is not installed."

        self.model_status.setText(text)
        self.clear_local_button.setEnabled(saved_path is not None)

    @Slot()
    def _choose_local_model(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose a faster-whisper model folder",
        )
        if not selected:
            return
        try:
            self.preferences.set_local_model_path(self._profile_key(), Path(selected))
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid model folder", str(exc))
            return
        self._refresh_status()

    @Slot()
    def _clear_local_model(self) -> None:
        self.preferences.clear_local_model_path(self._profile_key())
        self._refresh_status()

    @Slot()
    def _open_cache(self) -> None:
        cache = huggingface_cache_directory()
        if not cache.is_dir():
            QMessageBox.information(
                self,
                "Downloaded model folder",
                f"No downloaded model folder exists yet.\n\nExpected location:\n{cache}",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(cache)))


class MainWindow(QMainWindow):
    def __init__(self, preferences: ModelPreferences | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"CaptionMiner {__version__}")
        self.resize(840, 670)
        self._thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._close_after_cancel = False
        self._batch_started_at: float | None = None
        self._status_message = "Ready"
        self._model_preferences = preferences or ModelPreferences()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status_label)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        heading_row = QHBoxLayout()
        heading = QLabel("CaptionMiner")
        heading.setStyleSheet("font-size: 24px; font-weight: 700;")
        self.settings_button = QPushButton("Settings")
        heading_row.addWidget(heading)
        heading_row.addStretch(1)
        heading_row.addWidget(self.settings_button)
        layout.addLayout(heading_row)
        description = QLabel(
            "Drop exported clips below. CaptionMiner transcribes them locally and creates "
            "plain SRT subtitle files; styling remains in your video editor."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.media_list = MediaList()
        self.media_list.setToolTip("Drop local video or audio files here")
        self.media_list.paths_dropped.connect(self.add_paths)
        layout.addWidget(self.media_list)

        file_buttons = QHBoxLayout()
        self.add_button = QPushButton("Add files")
        self.remove_button = QPushButton("Remove selected")
        self.clear_button = QPushButton("Clear")
        file_buttons.addWidget(self.add_button)
        file_buttons.addWidget(self.remove_button)
        file_buttons.addWidget(self.clear_button)
        file_buttons.addStretch(1)
        layout.addLayout(file_buttons)

        settings_box = QGroupBox("Transcription")
        settings = QFormLayout(settings_box)
        self.profile_combo = QComboBox()
        for profile in MODEL_PROFILES.values():
            self.profile_combo.addItem(f"{profile.label} — {profile.model_name}", profile.key)
        self.profile_combo.setCurrentIndex(1)
        settings.addRow("Accuracy profile", self.profile_combo)

        self.language_combo = QComboBox()
        for label, code in COMMON_LANGUAGES:
            self.language_combo.addItem(label, code)
        settings.addRow("Spoken language", self.language_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Automatic", "auto")
        self.device_combo.addItem("NVIDIA CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        settings.addRow("Processing device", self.device_combo)

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "Optional names/terms: Snarkos, Veadotube, TwitchDownloader"
        )
        settings.addRow("Custom vocabulary", self.prompt_edit)

        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Empty = create each SRT beside its source clip")
        self.output_browse = QPushButton("Browse")
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_browse)
        settings.addRow("Output folder", output_row)

        self.overwrite_check = QCheckBox("Overwrite an existing same-named SRT")
        settings.addRow("", self.overwrite_check)
        layout.addWidget(settings_box)

        action_row = QHBoxLayout()
        self.start_button = QPushButton("Transcribe")
        self.start_button.setDefault(True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.cancel_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMinimumHeight(120)
        layout.addWidget(self.log)

        self.setCentralWidget(root)
        self.add_button.clicked.connect(self.choose_files)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button.clicked.connect(self.media_list.clear)
        self.output_browse.clicked.connect(self.choose_output_directory)
        self.start_button.clicked.connect(self.start_batch)
        self.cancel_button.clicked.connect(self.cancel_batch)
        self.settings_button.clicked.connect(self.open_settings)

    @Slot(list)
    def add_paths(self, paths: list[str]) -> None:
        existing = {self.media_list.item(i).text() for i in range(self.media_list.count())}
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.is_file() and str(path) not in existing:
                self.media_list.addItem(str(path))
                existing.add(str(path))

    @Slot()
    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select video or audio files",
            "",
            "Media files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mp3 *.wav "
            "*.m4a *.aac *.flac *.ogg *.opus);;All files (*)",
        )
        self.add_paths(files)

    @Slot()
    def remove_selected(self) -> None:
        for item in self.media_list.selectedItems():
            self.media_list.takeItem(self.media_list.row(item))

    @Slot()
    def choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose subtitle output folder")
        if selected:
            self.output_edit.setText(selected)

    def _set_running(self, running: bool) -> None:
        for widget in (
            self.add_button,
            self.remove_button,
            self.clear_button,
            self.start_button,
            self.profile_combo,
            self.language_combo,
            self.device_combo,
            self.prompt_edit,
            self.output_edit,
            self.output_browse,
            self.overwrite_check,
            self.media_list,
            self.settings_button,
        ):
            widget.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _set_progress_value(self, value: int) -> None:
        if value == INDETERMINATE_PROGRESS:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setTextVisible(False)
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setValue(max(0, min(100, value)))

    def _set_status_message(self, message: str) -> None:
        self._status_message = message
        self._refresh_status_label()

    @Slot()
    def _refresh_status_label(self) -> None:
        message = self._status_message
        if self._batch_started_at is not None:
            elapsed = format_elapsed(time.monotonic() - self._batch_started_at)
            message = f"{message} • elapsed {elapsed}"
        self.status_label.setText(message)

    @Slot()
    def start_batch(self) -> None:
        files = [Path(self.media_list.item(i).text()) for i in range(self.media_list.count())]
        if not files:
            QMessageBox.information(self, "CaptionMiner", "Add at least one media file first.")
            return

        output_text = self.output_edit.text().strip()
        output_directory = Path(output_text).expanduser() if output_text else None
        if (
            output_directory is not None
            and output_directory.exists()
            and not output_directory.is_dir()
        ):
            QMessageBox.warning(self, "CaptionMiner", "The selected output path is not a folder.")
            return

        profile_key = str(self.profile_combo.currentData())
        model_selection = self._select_model_for_transcription(profile_key)
        if model_selection is None:
            return

        self.log.clear()
        self._batch_started_at = time.monotonic()
        self._set_progress_value(INDETERMINATE_PROGRESS)
        self._set_status_message("Starting...")
        self._status_timer.start()
        self._set_running(True)

        self._thread = QThread(self)
        self._worker = BatchWorker(
            files,
            profile=str(self.profile_combo.currentData()),
            language=self.language_combo.currentData(),
            device=str(self.device_combo.currentData()),
            initial_prompt=self.prompt_edit.text().strip() or None,
            output_directory=output_directory,
            overwrite=self.overwrite_check.isChecked(),
            model_reference=model_selection.reference,
            local_files_only=model_selection.local_files_only,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.log.connect(self.log.appendPlainText)
        self._worker.completed.connect(self.on_completed)
        self._worker.completed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _select_model_for_transcription(self, profile_key: str) -> ModelSelection | None:
        profile = MODEL_PROFILES[profile_key]

        while True:
            lookup = resolve_installed_model(
                profile_key,
                profile.model_name,
                self._model_preferences,
            )
            if lookup.invalid_local_reason:
                QMessageBox.warning(
                    self,
                    "Local model unavailable",
                    lookup.invalid_local_reason
                    + "\n\nThe saved local selection will be removed. You can choose it again "
                    "after fixing the folder.",
                )
                self._model_preferences.clear_local_model_path(profile_key)
                if lookup.selection is None:
                    continue
            if lookup.selection is not None:
                return lookup.selection

            policy = self._model_preferences.download_policy
            if policy is DownloadPolicy.ALLOW:
                return ModelSelection(
                    reference=profile.model_name,
                    location=huggingface_cache_directory(),
                    source="download",
                    local_files_only=False,
                )
            if policy is DownloadPolicy.DENY:
                action = self._show_downloads_disabled(profile_key)
                if action == "settings":
                    self.open_settings()
                    continue
                if action == "local":
                    selection = self._choose_local_model_for_profile(profile_key)
                    if selection is not None:
                        return selection
                return None

            effect = apply_download_consent_action(
                self._model_preferences,
                self._ask_for_download_consent(profile_key),
            )
            if effect.allow_once:
                return ModelSelection(
                    reference=profile.model_name,
                    location=huggingface_cache_directory(),
                    source="download",
                    local_files_only=False,
                )
            if effect.choose_local:
                selection = self._choose_local_model_for_profile(profile_key)
                if selection is not None:
                    return selection
                return None
            return None

    def _ask_for_download_consent(self, profile_key: str) -> DownloadConsentAction:
        profile = MODEL_PROFILES[profile_key]
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Question)
        message.setWindowTitle("Speech-recognition model required")
        message.setText(f"The {profile.label} model is not installed.")
        message.setInformativeText(
            "CaptionMiner can download it once from Hugging Face and reuse it later. "
            "Your video or audio will not be uploaded.\n\n"
            f"Model: {profile.model_name}\n"
            f"Download location: {huggingface_cache_directory()}"
        )
        download_button = message.addButton("Download model", QMessageBox.AcceptRole)
        local_button = message.addButton("Choose local model folder", QMessageBox.ActionRole)
        deny_button = message.addButton("No", QMessageBox.RejectRole)
        message.setDefaultButton(download_button)
        message.exec()

        clicked = message.clickedButton()
        if clicked is download_button:
            return DownloadConsentAction.DOWNLOAD
        if clicked is local_button:
            return DownloadConsentAction.LOCAL
        if clicked is deny_button:
            return DownloadConsentAction.DENY
        return DownloadConsentAction.CANCEL

    def _show_downloads_disabled(self, profile_key: str) -> str:
        profile = MODEL_PROFILES[profile_key]
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Model downloads are disabled")
        message.setText(f"The {profile.label} model is not installed.")
        message.setInformativeText(
            "Choose a local model folder or change the model-download preference in Settings."
        )
        local_button = message.addButton("Choose local model folder", QMessageBox.ActionRole)
        settings_button = message.addButton("Open Settings", QMessageBox.ActionRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(local_button)
        message.exec()

        clicked = message.clickedButton()
        if clicked is local_button:
            return "local"
        if clicked is settings_button:
            return "settings"
        if clicked is cancel_button:
            return "cancel"
        return "cancel"

    def _choose_local_model_for_profile(self, profile_key: str) -> ModelSelection | None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose a faster-whisper model folder",
        )
        if not selected:
            return None
        try:
            self._model_preferences.set_local_model_path(profile_key, Path(selected))
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid model folder", str(exc))
            return None
        lookup = resolve_installed_model(
            profile_key,
            MODEL_PROFILES[profile_key].model_name,
            self._model_preferences,
        )
        return lookup.selection

    @Slot()
    def open_settings(self) -> None:
        ModelSettingsDialog(
            self._model_preferences,
            self,
            initial_profile_key=str(self.profile_combo.currentData()),
        ).exec()

    @Slot()
    def cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self._set_status_message("Cancelling after the current safe checkpoint...")

    @Slot(int, str)
    def on_progress(self, value: int, message: str) -> None:
        self._set_progress_value(value)
        self._set_status_message(message)

    @Slot(int, int, bool)
    def on_completed(self, completed: int, failed: int, cancelled: bool) -> None:
        self._set_running(False)
        self._status_timer.stop()
        if cancelled:
            message = f"Cancelled. Created {completed} subtitle file(s); {failed} failed."
        else:
            message = f"Finished. Created {completed} subtitle file(s); {failed} failed."
        self._set_status_message(message)
        self._batch_started_at = None
        self.log.appendPlainText(message)
        self._worker = None
        self._thread = None
        if self._close_after_cancel:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._worker is None:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "CaptionMiner is running",
            "Cancel the current batch and close after processing reaches a safe checkpoint?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._close_after_cancel = True
            self._worker.cancel()
            self._set_status_message("Cancelling after the current safe checkpoint...")
            event.ignore()
        else:
            event.ignore()


def run_gui() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("CaptionMiner")
    app.setOrganizationName("CaptionMiner")
    window = MainWindow()
    window.show()
    return app.exec()
