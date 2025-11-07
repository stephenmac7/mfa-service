import shutil
import tempfile
import logging
from contextlib import asynccontextmanager
import wave
import contextlib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse

import tgt
from kalpy.aligner import KalpyAligner
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import LexiconCompiler
from kalpy.utterance import Segment
from kalpy.utterance import Utterance as KalpyUtterance

# MFA models and utilities
from montreal_forced_aligner.models import AcousticModel, G2PModel
from montreal_forced_aligner.online.alignment import tokenize_utterance_text
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer
from montreal_forced_aligner.data import (
    Language,
    OOV_WORD,
    LAUGHTER_WORD,
    BRACKETED_WORD,
    CUTOFF_WORD,
)
from montreal_forced_aligner.dictionary.mixins import (
    DEFAULT_BRACKETS,
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    DEFAULT_WORD_BREAK_MARKERS,
)

from montreal_forced_aligner.command_line.utils import validate_dictionary, validate_g2p_model


# --- Model Configuration ---
# ACOUSTIC_MODEL_NAME = 'english_us_arpa'
# DICTIONARY_NAME = 'english_us_arpa'
# G2P_MODEL_NAME = 'english_us_arpa' # Optional, for OOV words (currently broken due to bug in MFA)
ACOUSTIC_MODEL_NAME = "english_mfa"
DICTIONARY_NAME = "english_us_mfa"
# G2P_MODEL_NAME = 'english_us_mfa'  # Optional, for OOV words (currently broken due to bug in MFA)
G2P_MODEL_NAME = None
# ---------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- Lib --
def parse_textgrid_to_json(textgrid_path: str, symbols_to_filter: set) -> dict:
    """Parses a TextGrid file using tgt and converts it into a JSON-serializable dictionary."""
    if tgt is None:
        raise ImportError("The 'tgt' library is required to parse output.")

    tg = tgt.io.read_textgrid(textgrid_path)
    response_data = {}
    for tier in tg.tiers:
        if not isinstance(tier, tgt.IntervalTier):
            continue

        intervals_data = []
        for interval in tier.intervals:
            text = interval.text
            if not text:
                continue
            if text in symbols_to_filter:
                continue
            if text.startswith("#"):  # Filters disambiguation phones like #0, #1
                continue

            if interval.text:
                intervals_data.append(
                    {
                        "start": round(interval.start_time, 4),
                        "end": round(interval.end_time, 4),
                        "content": text,
                    }
                )
        response_data[tier.name] = intervals_data
    return response_data


# -- FastAPI App --


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all necessary MFA/Kalpy models into app.state on server startup.
    """
    logger.info("Loading models... This may take a moment.")

    # 1. Load Acoustic Model
    try:
        acoustic_model_path = AcousticModel.get_pretrained_path(ACOUSTIC_MODEL_NAME)
        if acoustic_model_path is None:
            raise FileNotFoundError(f"Acoustic model '{ACOUSTIC_MODEL_NAME}' not found.")
        app.state.acoustic_model = AcousticModel(acoustic_model_path)
        logger.info(f"Loaded acoustic model: {acoustic_model_path}")
    except Exception as e:
        logger.error(f"Failed to load acoustic model: {e}", exc_info=True)
        raise

    # 2. Load Dictionary and Compile Lexicon
    try:
        dictionary_path = validate_dictionary(None, None, DICTIONARY_NAME)

        if dictionary_path is None:
            raise FileNotFoundError(f"Dictionary '{DICTIONARY_NAME}' not found.")

        app.state.lexicon_compiler = LexiconCompiler(
            disambiguation=False,
            silence_probability=app.state.acoustic_model.parameters["silence_probability"],
            initial_silence_probability=app.state.acoustic_model.parameters[
                "initial_silence_probability"
            ],
            final_silence_correction=app.state.acoustic_model.parameters[
                "final_silence_correction"
            ],
            final_non_silence_correction=app.state.acoustic_model.parameters[
                "final_non_silence_correction"
            ],
            silence_phone=app.state.acoustic_model.parameters["optional_silence_phone"],
            oov_phone=app.state.acoustic_model.parameters["oov_phone"],
            position_dependent_phones=app.state.acoustic_model.parameters[
                "position_dependent_phones"
            ],
            phones=app.state.acoustic_model.parameters["non_silence_phones"],
            ignore_case=True,
        )
        app.state.lexicon_compiler.load_pronunciations(dictionary_path)
        logger.info(f"Loaded and compiled dictionary: {dictionary_path}")
        # Record initial lexicon checksum for later change detection
        app.state.lexicon_checksum = app.state.lexicon_compiler.word_table.checksum()

    except Exception as e:
        logger.error(f"Failed to load dictionary: {e}", exc_info=True)
        raise

    # 3. Load G2P Model (for OOV words)
    if G2P_MODEL_NAME is None:
        app.state.g2p_model = None
        logger.info("Continuing without G2P model (no OOV handling).")
    else:
        g2p_model_path = G2PModel.get_pretrained_path(G2P_MODEL_NAME)
        if g2p_model_path is None:
            raise FileNotFoundError(f"G2P model '{G2P_MODEL_NAME}' not found.")
        g2p_model_path = validate_g2p_model(None, None, g2p_model_path)
        app.state.g2p_model = G2PModel(g2p_model_path)
        logger.info(f"Loaded G2P model: {g2p_model_path}")

    # 4. Determine language and set tokenizer
    am_params = app.state.acoustic_model.parameters
    lang = Language[am_params.get("language", "unknown")]
    logger.info(f"Using language from acoustic model: {lang}")

    if lang is Language.unknown:
        logger.info("Using SimpleTokenizer (language unknown).")
        app.state.tokenizer = SimpleTokenizer(
            word_table=app.state.lexicon_compiler.word_table,
            word_break_markers=DEFAULT_WORD_BREAK_MARKERS,
            punctuation=DEFAULT_PUNCTUATION,
            clitic_markers=DEFAULT_CLITIC_MARKERS,
            compound_markers=DEFAULT_COMPOUND_MARKERS,
            brackets=DEFAULT_BRACKETS,
            laughter_word=LAUGHTER_WORD,
            oov_word=OOV_WORD,
            bracketed_word=BRACKETED_WORD,
            cutoff_word=CUTOFF_WORD,
            ignore_case=True,
        )
    else:
        # this doesn't actually work, but is what the CLI MFA code does
        logger.info(f"Using SpaCy tokenizer for language: {lang}")
        app.state.tokenizer = generate_language_tokenizer(lang)

    app.state.language = lang

    # 5. Create the Kalpy Aligner
    app.state.kalpy_aligner = KalpyAligner(
        app.state.acoustic_model,
        app.state.lexicon_compiler,
        beam=10,
        retry_beam=40,
        acoustic_scale=0.1,
        transition_scale=1.0,
        self_loop_scale=0.1,
    )
    am_params = app.state.acoustic_model.parameters
    app.state.filter_symbols = {
        am_params["optional_silence_phone"],  # Ex: 'sil'
        am_params["oov_phone"],  # Ex: 'spn'
        am_params.get("silence_word", "<eps>"),  # Ex: '<eps>'
    }
    if "other_noise_phone" in am_params and am_params["other_noise_phone"]:
        app.state.filter_symbols.add(am_params["other_noise_phone"])

    logger.info("Models loaded and aligner is ready.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Kalpy/MFA Aligner API is running. POST to /align to process files."}


@app.post("/align", response_class=JSONResponse)
def align_audio(
    request: Request,
    audio: UploadFile = File(..., description="A WAV audio file."),
    transcript: str = Form(..., description="The transcript corresponding to the audio."),
):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a .wav audio file."
        )

    # Get pre-loaded models from app.state
    acoustic_model = request.app.state.acoustic_model
    lexicon_compiler = request.app.state.lexicon_compiler
    g2p_model = request.app.state.g2p_model
    tokenizer = request.app.state.tokenizer
    kalpy_aligner = request.app.state.kalpy_aligner

    # We must save the audio to a temporary file, as kalpy needs a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        try:
            shutil.copyfileobj(audio.file, temp_audio_file)
            temp_audio_file.flush()  # Ensure all data is written
            temp_audio_path = temp_audio_file.name
            logger.info(f"Saved audio to temporary file: {temp_audio_path}")

            # 1. Get audio file duration using the wave module
            try:
                with contextlib.closing(wave.open(temp_audio_path, "r")) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                    logger.info(f"Audio duration: {duration}s")
            except wave.Error as e:
                logger.error(f"Could not read WAV file info: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid or corrupted WAV file: {e}")

            # 2. Create a Kalpy Segment for the whole file
            segment = Segment(temp_audio_path, 0, duration, 0)

            # 3. Tokenize the transcript text
            logger.info(f"Original transcript: {transcript}")
            pre_sig = request.app.state.lexicon_checksum
            normalized_text = tokenize_utterance_text(
                text=transcript,
                lexicon_compiler=lexicon_compiler,
                tokenizer=tokenizer,
                g2p_model=g2p_model,
                language=request.app.state.language,
            )
            logger.info(f"Normalized transcript: {' '.join(normalized_text)}")

            # If G2P added entries to the lexicon, rebuild the aligner
            # Failing to rebuild the aligner causes alignment to segfault
            request.app.state.lexicon_checksum = lexicon_compiler.word_table.checksum()
            if request.app.state.lexicon_checksum != pre_sig:
                request.app.state.kalpy_aligner = KalpyAligner(
                    acoustic_model,
                    lexicon_compiler,
                    beam=10,
                    retry_beam=40,
                    acoustic_scale=0.1,
                    transition_scale=1.0,
                    self_loop_scale=0.1,
                )
                kalpy_aligner = request.app.state.kalpy_aligner

            # 4. Create a Kalpy Utterance
            utt = KalpyUtterance(segment, normalized_text)

            # 5. Generate Features (MFCCs)
            utt.generate_mfccs(acoustic_model.mfcc_computer)

            # 6. Compute CMVN (feature normalization)
            cmvn_computer = CmvnComputer()
            cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs])
            utt.apply_cmvn(cmvn)

            # 7. Align
            logger.info("Starting alignment...")
            ctm = kalpy_aligner.align_utterance(utt)
            logger.info("Alignment finished.")

            # 8. Export to a temporary TextGrid file
            with tempfile.NamedTemporaryFile(suffix=".TextGrid", delete=True) as temp_tg_file:
                tg_path = temp_tg_file.name
                ctm.export_textgrid(tg_path, file_duration=duration, output_format="long_textgrid")

                # 9. Parse the TextGrid to JSON
                logger.info(f"Parsing TextGrid from: {tg_path}")
                json_output = parse_textgrid_to_json(tg_path, app.state.filter_symbols)

            return JSONResponse(content=json_output)

        except Exception as e:
            logger.error(f"An error occurred during alignment: {e}", exc_info=True)
            detail = f"An internal error occurred: {str(e)}"
            raise HTTPException(status_code=500, detail=detail)
