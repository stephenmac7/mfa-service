# MFA Service

A FastAPI-based web service for forced alignment of audio and text using Montreal Forced Aligner (MFA) and Kalpy.

## Features

- RESTful API for audio-transcript alignment
- Returns alignment results as JSON with word and phone-level timing
- Currently English-only (hardcodes pretrained English US ARPA)

## Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:stephenmac7/mfa-service.git
   cd mfa-service
   ```

2. Create the conda environment from the environment file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate mfa-service
   ```

4. Download the required MFA models:
   ```bash
   mfa model download acoustic english_us_arpa
   mfa model download dictionary english_us_arpa
   ```

### Running the Server

1. Activate the conda environment (if not already activated):
   ```bash
   conda activate mfa-service
   ```

2. Start the FastAPI server:
   ```bash
   fastapi dev main.py --port 8001
   ```

3. The API will be available at `http://localhost:8001`
   - Interactive API docs: `http://localhost:8001/docs`
   - Alternative docs: `http://localhost:8001/redoc`

## Usage

### API Endpoint

**POST /align**

Upload a WAV file and transcript to get alignment results.

**Parameters:**
- `audio` (file): WAV audio file
- `transcript` (form field): Text transcript of the audio

**Example using curl:**
```bash
curl -X POST "http://localhost:8001/align" \
  -F "audio=@your_audio.wav" \
  -F "transcript=Your transcript text here"
```

**Example using Python:**
```python
import requests

files = {'audio': open('your_audio.wav', 'rb')}
data = {'transcript': 'Your transcript text here'}
response = requests.post('http://localhost:8001/align', files=files, data=data)
print(response.json())
```

## Development

### Updating Dependencies

If you add new dependencies, update the `environment.yml` file and recreate the environment:

```bash
conda env update -f environment.yml --prune
```