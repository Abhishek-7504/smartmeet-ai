#!/usr/bin/env python3
import os
import shutil
import click
import json
from mutagen import File
import wave

WORKDIR = "workspace"

@click.command()
@click.argument("src", type=click.Path(exists=True))
def ingest(src):
    os.makedirs(WORKDIR, exist_ok=True)
    dest = os.path.join(WORKDIR, os.path.basename(src))
    shutil.copy(src, dest)

    audio = File(dest)
    duration = None

    if audio and audio.info and hasattr(audio.info, 'length'):
        duration = audio.info.length
    else:
        # fallback to wave module for wav files
        try:
            with wave.open(dest, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
        except wave.Error:
            duration = None

    if duration is not None:
        duration_str = f"{duration:.3f}"
    else:
        duration_str = "Unknown"

    size_str = str(os.path.getsize(dest))

    metadata = {
        "duration": duration_str,
        "size": size_str,
    }

    click.echo("Ingested to: " + dest)
    click.echo("Metadata:")
    click.echo(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    ingest()
