# Worker Notes (per dev)

## Ruolo del worker
Il worker e **solo un esecutore**: non gestisce login, non e source of truth.  
Tutto il vero stato (utenti, libri, job, storage) vive nel Central.

## Contratto con il Central
Il worker parla solo con endpoint interni del Central usando `X-Worker-Token`:
- `GET /internal/jobs/{id}` (info job + parametri)
- `GET /internal/jobs/{id}/input` (download file)
- `POST /internal/jobs/{id}/heartbeat`
- `POST /internal/jobs/{id}/artifacts`
- `POST /internal/jobs/{id}/log`
- `POST /internal/jobs/{id}/complete`

## Env richieste
Queste variabili sono passate dal Central al pod:
- `WORKER_JOB_ID`
- `CENTRAL_BASE_URL`
- `WORKER_SHARED_TOKEN`
- `PYTHONPATH=/workspace/Chatterblez_FINITIO`

## Pipeline (macro-step)
1) Download input dal Central  
2) Split capitoli / frasi  
3) TTS Azzurra/CSM  
4) ffmpeg (m4a)  
5) Whisper (srt)  
6) Upload risultati + log al Central  
7) Notifica completamento

## DB locale
Esiste un sqlite locale **solo per compatibilita** con la pipeline legacy.  
Non contiene credenziali reali e non fa login.

## Stati job
- `queued` → creato dal Central  
- `running` → primo heartbeat o avvio worker  
- `success/failed/canceled` → finalizzazione dal Central

## Runpod note
- Il volume non deve essere montato su `/workspace` (coprirebbe l'immagine).
- Usare `RUNPOD_VOLUME_MOUNT_PATH=/runpod-volume`.

## Workaround temporaneo
C'e una patch al volo nel Central per sistemare `timeout` in `CentralClient`.  
Da rimuovere dopo rebuild dell'immagine worker.
