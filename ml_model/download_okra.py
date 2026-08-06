"""
Resumable parallel downloader for Okra DiseaseNet (Mendeley nh7zk4hv8z.1).

Improvements over v1:
  * requests.Session (keep-alive connection reuse -> far fewer TLS handshakes)
  * atomic writes (temp file + rename) so interrupted files never block resume
  * jittered retry backoff; files already present with matching size are skipped
"""
import os
import sys
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

META = r'C:\Users\amuly\AppData\Local\Temp\opencode\okra_dataset_meta.json'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'dataset', 'okra', 'raw')
WORKERS = 16
RETRIES = 6
SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'Mozilla/5.0 leaf-lenz'})


def download_one(item):
    fname = item['filename']
    url = item['content_details']['download_url']
    expected = item.get('size', -1)
    cls = fname.split('_')[0]
    outdir = os.path.join(RAW, cls)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, fname)
    if os.path.exists(outpath) and os.path.getsize(outpath) == expected:
        return (fname, 'skip')
    tmp = outpath + '.part'
    for attempt in range(1, RETRIES + 1):
        try:
            with SESSION.get(url, timeout=(30, 300), stream=True) as r:
                r.raise_for_status()
                size = 0
                with open(tmp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            size += len(chunk)
            if size != expected:
                raise ValueError(f'size mismatch {size} != {expected}')
            os.replace(tmp, outpath)
            return (fname, 'ok')
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            if attempt == RETRIES:
                return (fname, f'fail:{type(e).__name__}:{e}')
            time.sleep(random.uniform(1, 3) * attempt)
    return (fname, 'fail')


def main():
    with open(META, encoding='utf-8-sig') as f:
        meta = json.load(f)
    files = [f for f in meta['files']
             if f.get('content_details', {}).get('content_type') == 'image/jpeg']
    os.makedirs(RAW, exist_ok=True)
    print(f'total jpg to download: {len(files)}', flush=True)
    results = {'ok': 0, 'skip': 0, 'fail': []}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download_one, item) for item in files]
        for i, fut in enumerate(as_completed(futs), 1):
            fname, status = fut.result()
            if status == 'ok':
                results['ok'] += 1
            elif status == 'skip':
                results['skip'] += 1
            else:
                results['fail'].append((fname, status))
            if i % 100 == 0 or i == len(files):
                el = time.time() - t0
                print(f'progress {i}/{len(files)} ok={results["ok"]} skip={results["skip"]} '
                      f'fail={len(results["fail"])} elapsed={el/60:.1f}min', flush=True)
    print('DONE', results, flush=True)


if __name__ == '__main__':
    main()
