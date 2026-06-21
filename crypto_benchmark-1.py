"""
=============================================================================
Project Title:
Comparative Analysis of Some Encryption and Decryption Algorithms On Text 
File Security.

Author:
Iniobong Peace Effiong
=============================================================================
"""
"""
=============================================================================
Cryptographic Algorithm Performance Evaluation
Compares: AES-256 (GCM), RSA-2048 (Hybrid), ChaCha20, ECC-256 (ECIES)
=============================================================================
"""

import os
import time
import random
import string
import tracemalloc
import csv
import statistics
import sys
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    print("[INSTALL] Installing pandas …")
    os.system(f"{sys.executable} -m pip install pandas --quiet")
    import pandas as pd

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("[INSTALL] Installing cryptography …")
    os.system(f"{sys.executable} -m pip install cryptography --quiet")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED   = 42
FILE_SIZES_KB = [10, 100, 1_024, 5_120, 10_240]   # 10 KB → 10 MB
NUM_RUNS      = 10
OUTPUT_DIR    = Path("crypto_output")
FILES_DIR     = OUTPUT_DIR / "synthetic_files"
RAW_CSV       = OUTPUT_DIR / "raw_results.csv"
SUMMARY_CSV   = OUTPUT_DIR / "summary_results.csv"

ALGORITHMS    = ["AES-256-GCM", "RSA-2048-Hybrid", "ChaCha20-Poly1305", "ECC-256-ECIES"]

RAW_COLUMNS = [
    "Algorithm", "File_Size_KB", "Run",
    "KeyGen_Time_s", "KeyGen_Memory_KB",
    "Enc_Time_s", "Enc_Memory_KB", "Ciphertext_Size_B", "Expansion_Ratio",
    "Dec_Time_s", "Dec_Memory_KB",
    "Enc_Throughput_MBps", "Dec_Throughput_MBps",
    "Decryption_Valid",
]

# =============================================================================
# HELPERS
# =============================================================================

def _measure(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), returning (result, elapsed_s, peak_memory_KB)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / 1024  # bytes → KB


def _progress(msg: str):
    print(f"  {msg}", flush=True)


# =============================================================================
# 1. SYNTHETIC FILE GENERATION
# =============================================================================

def generate_synthetic_files(seed: int = RANDOM_SEED) -> dict[int, Path]:
    """
    Create one text file per requested size (KB).
    Content is reproducible via a fixed random seed.
    Returns {size_kb: file_path}.
    """
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    vocab = string.ascii_letters + string.digits + " \n"

    paths = {}
    for kb in FILE_SIZES_KB:
        fpath = FILES_DIR / f"synthetic_{kb}KB.txt"
        if not fpath.exists():
            target_bytes = kb * 1024
            content = "".join(rng.choices(vocab, k=target_bytes))
            fpath.write_text(content, encoding="utf-8")
        paths[kb] = fpath
        _progress(f"File ready: {fpath.name} ({kb} KB)")
    return paths


# =============================================================================
# 2. KEY GENERATION
# =============================================================================

def _gen_aes_key():
    return os.urandom(32)                       # 256-bit key


def _gen_rsa_keypair():
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key


def _gen_chacha20_key():
    return os.urandom(32)                       # 256-bit key (nonce per-message)


def _gen_ecc_keypair():
    """Generate an ECC-256 (SECP256R1 / P-256, ~128-bit security) keypair."""
    private_key = ec.generate_private_key(ec.SECP256R1(), backend=default_backend())
    public_key  = private_key.public_key()
    return private_key, public_key


KEYGEN_FNS = {
    "AES-256-GCM":        _gen_aes_key,
    "RSA-2048-Hybrid":    _gen_rsa_keypair,
    "ChaCha20-Poly1305":  _gen_chacha20_key,
    "ECC-256-ECIES":      _gen_ecc_keypair,
}


def generate_key(algorithm: str):
    """Return (key_material, elapsed_s, peak_KB)."""
    fn = KEYGEN_FNS[algorithm]
    return _measure(fn)


# =============================================================================
# 3 & 4. ENCRYPTION / DECRYPTION
# =============================================================================

# ── AES-256-GCM ──────────────────────────────────────────────────────────────

def encrypt_aes(plaintext: bytes, key: bytes) -> bytes:
    """AES-256-GCM: prepend 12-byte nonce to ciphertext."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ct


def decrypt_aes(ciphertext: bytes, key: bytes) -> bytes:
    nonce, ct = ciphertext[:12], ciphertext[12:]
    return AESGCM(key).decrypt(nonce, ct, None)


# ── RSA-2048 Hybrid (RSA wraps an ephemeral AES key) ─────────────────────────

def encrypt_rsa_hybrid(plaintext: bytes, public_key) -> bytes:
    """
    Hybrid encryption:
      1. Generate ephemeral AES-256-GCM key.
      2. Encrypt plaintext with AES.
      3. Encrypt the AES key with RSA-OAEP.
      4. Packet = [4-byte wrapped-key-len][wrapped-key][nonce+ciphertext]
    """
    aes_key   = os.urandom(32)
    nonce     = os.urandom(12)
    ct_body   = AESGCM(aes_key).encrypt(nonce, plaintext, None)

    wrapped   = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    wk_len = len(wrapped).to_bytes(4, "big")
    return wk_len + wrapped + nonce + ct_body


def decrypt_rsa_hybrid(ciphertext: bytes, private_key) -> bytes:
    wk_len   = int.from_bytes(ciphertext[:4], "big")
    wrapped  = ciphertext[4 : 4 + wk_len]
    rest     = ciphertext[4 + wk_len:]
    nonce, ct_body = rest[:12], rest[12:]

    aes_key = private_key.decrypt(
        wrapped,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return AESGCM(aes_key).decrypt(nonce, ct_body, None)


# ── ChaCha20-Poly1305 ────────────────────────────────────────────────────────

def encrypt_chacha20(plaintext: bytes, key: bytes) -> bytes:
    """ChaCha20-Poly1305: prepend 12-byte nonce."""
    nonce = os.urandom(12)
    ct    = ChaCha20Poly1305(key).encrypt(nonce, plaintext, None)
    return nonce + ct


def decrypt_chacha20(ciphertext: bytes, key: bytes) -> bytes:
    nonce, ct = ciphertext[:12], ciphertext[12:]
    return ChaCha20Poly1305(key).decrypt(nonce, ct, None)


# ── ECC-256 ECIES (Elliptic Curve Integrated Encryption Scheme) ─────────────
#
#   Curve   : SECP256R1 / P-256  (≈128-bit symmetric security, "ECC-256")
#   KEM     : Ephemeral-static ECDH (ephemeral sender key + recipient pubkey)
#   KDF     : HKDF-SHA256 → 32-byte AES-256 key
#   DEM     : AES-256-GCM (authenticated encryption)
#
#   Packet format = [1-byte eph_pub_len][eph_pub (uncompressed point)]
#                   [12-byte nonce][AES-GCM ciphertext+tag]
#
#   This is the standard ECIES construction: a fresh ephemeral EC keypair is
#   generated for every encryption, an ECDH shared secret is derived against
#   the recipient's static public key, and that secret is stretched via HKDF
#   into a symmetric key that drives AES-256-GCM. Only the recipient holding
#   the static private key can reproduce the shared secret and decrypt.

def _ecies_derive_key(shared_secret: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,                 # 256-bit AES key
        salt=None,
        info=b"ECIES-ECC256-AESGCM",
        backend=default_backend(),
    ).derive(shared_secret)


def encrypt_ecc(plaintext: bytes, public_key) -> bytes:
    # 1. Ephemeral keypair for this message only.
    eph_private = ec.generate_private_key(ec.SECP256R1(), backend=default_backend())
    eph_public_bytes = eph_private.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )

    # 2. ECDH shared secret with recipient's static public key.
    shared_secret = eph_private.exchange(ec.ECDH(), public_key)

    # 3. Derive AES-256 key via HKDF.
    aes_key = _ecies_derive_key(shared_secret)

    # 4. Symmetric encryption (DEM) with AES-256-GCM.
    nonce   = os.urandom(12)
    ct_body = AESGCM(aes_key).encrypt(nonce, plaintext, None)

    pub_len = len(eph_public_bytes).to_bytes(1, "big")
    return pub_len + eph_public_bytes + nonce + ct_body


def decrypt_ecc(ciphertext: bytes, private_key) -> bytes:
    pub_len = ciphertext[0]
    eph_public_bytes = ciphertext[1 : 1 + pub_len]
    rest = ciphertext[1 + pub_len :]
    nonce, ct_body = rest[:12], rest[12:]

    eph_public = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), eph_public_bytes
    )
    shared_secret = private_key.exchange(ec.ECDH(), eph_public)
    aes_key = _ecies_derive_key(shared_secret)
    return AESGCM(aes_key).decrypt(nonce, ct_body, None)


# ── Dispatch tables ───────────────────────────────────────────────────────────

ENC_FNS = {
    "AES-256-GCM":       encrypt_aes,
    "RSA-2048-Hybrid":   encrypt_rsa_hybrid,
    "ChaCha20-Poly1305": encrypt_chacha20,
    "ECC-256-ECIES":     encrypt_ecc,
}
DEC_FNS = {
    "AES-256-GCM":       decrypt_aes,
    "RSA-2048-Hybrid":   decrypt_rsa_hybrid,
    "ChaCha20-Poly1305": decrypt_chacha20,
    "ECC-256-ECIES":     decrypt_ecc,
}


def encrypt_file(algorithm: str, plaintext: bytes, key_material):
    """Return (ciphertext, elapsed_s, peak_KB)."""
    fn = ENC_FNS[algorithm]
    return _measure(fn, plaintext, key_material)


def decrypt_file(algorithm: str, ciphertext: bytes, key_material):
    """Return (plaintext, elapsed_s, peak_KB)."""
    fn = DEC_FNS[algorithm]
    return _measure(fn, ciphertext, key_material)


# =============================================================================
# 5 & 6. EXPERIMENT RUNNER
# =============================================================================

def _enc_key(algorithm: str, key_material):
    """Return the encryption-side key from key_material tuple/object."""
    if algorithm in ("RSA-2048-Hybrid", "ECC-256-ECIES"):
        return key_material[1]   # public key
    return key_material


def _dec_key(algorithm: str, key_material):
    """Return the decryption-side key from key_material tuple/object."""
    if algorithm in ("RSA-2048-Hybrid", "ECC-256-ECIES"):
        return key_material[0]   # private key
    return key_material


def run_experiment(file_paths: dict) -> list[dict]:
    """
    For every (algorithm, file_size, run) triple:
      - generate key
      - encrypt
      - decrypt
      - record metrics
    Returns a list of row dicts.
    """
    rows = []
    total = len(ALGORITHMS) * len(FILE_SIZES_KB) * NUM_RUNS
    done  = 0

    for algo in ALGORITHMS:
        for kb, fpath in file_paths.items():
            plaintext = fpath.read_bytes()
            file_mb   = len(plaintext) / (1024 ** 2)

            print(f"\n[{algo}] {kb} KB", flush=True)

            for run in range(1, NUM_RUNS + 1):
                done += 1
                pct  = done / total * 100

                # ── Key generation ────────────────────────────────────────
                key_material, kg_time, kg_mem = generate_key(algo)

                # ── Encryption ────────────────────────────────────────────
                enc_key = _enc_key(algo, key_material)
                ciphertext, enc_time, enc_mem = encrypt_file(algo, plaintext, enc_key)
                ct_size = len(ciphertext)
                expansion = ct_size / len(plaintext)
                enc_tput  = file_mb / enc_time if enc_time > 0 else float("inf")

                # ── Decryption ────────────────────────────────────────────
                dec_key_ = _dec_key(algo, key_material)
                try:
                    recovered, dec_time, dec_mem = decrypt_file(algo, ciphertext, dec_key_)
                    valid = (recovered == plaintext)
                except Exception as exc:
                    _progress(f"  ⚠ Decryption error run {run}: {exc}")
                    dec_time, dec_mem, valid = 0.0, 0.0, False
                dec_tput = file_mb / dec_time if dec_time > 0 else float("inf")

                rows.append({
                    "Algorithm":        algo,
                    "File_Size_KB":     kb,
                    "Run":              run,
                    "KeyGen_Time_s":    round(kg_time,  6),
                    "KeyGen_Memory_KB": round(kg_mem,   3),
                    "Enc_Time_s":       round(enc_time, 6),
                    "Enc_Memory_KB":    round(enc_mem,  3),
                    "Ciphertext_Size_B":ct_size,
                    "Expansion_Ratio":  round(expansion, 6),
                    "Dec_Time_s":       round(dec_time, 6),
                    "Dec_Memory_KB":    round(dec_mem,  3),
                    "Enc_Throughput_MBps": round(enc_tput, 4),
                    "Dec_Throughput_MBps": round(dec_tput, 4),
                    "Decryption_Valid": valid,
                })

                _progress(
                    f"  Run {run:02}/{NUM_RUNS} | "
                    f"Enc {enc_time*1000:7.2f} ms | "
                    f"Dec {dec_time*1000:7.2f} ms | "
                    f"✓ {valid} | {pct:5.1f}% done"
                )

    return rows


# =============================================================================
# 7 & 8. DATA STORAGE & STATISTICS
# =============================================================================

def save_raw(rows: list[dict]) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df.to_csv(RAW_CSV, index=False)
    print(f"\n[CSV] Raw results → {RAW_CSV}")
    return df


NUMERIC_METRICS = [
    "KeyGen_Time_s", "KeyGen_Memory_KB",
    "Enc_Time_s", "Enc_Memory_KB",
    "Dec_Time_s", "Dec_Memory_KB",
    "Ciphertext_Size_B", "Expansion_Ratio",
    "Enc_Throughput_MBps", "Dec_Throughput_MBps",
]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (Algorithm, File_Size_KB), compute mean ± std for every metric."""
    groups = df.groupby(["Algorithm", "File_Size_KB"])

    summary_rows = []
    for (algo, kb), grp in groups:
        row = {"Algorithm": algo, "File_Size_KB": kb}
        for col in NUMERIC_METRICS:
            vals = grp[col].dropna().tolist()
            row[f"{col}_mean"] = round(statistics.mean(vals), 6) if vals else None
            row[f"{col}_std"]  = round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[CSV] Summary results → {SUMMARY_CSV}")
    return summary_df


# =============================================================================
# 12. CONSOLE SUMMARY TABLE
# =============================================================================

def print_summary_table(summary_df: pd.DataFrame):
    """Print a human-readable pivot of key metrics."""
    print("\n" + "=" * 90)
    print(" PERFORMANCE SUMMARY  (mean ± std over 10 runs)")
    print("=" * 90)

    metrics_to_show = [
        ("Enc_Time_s",         "Enc Time (ms)",   1000, 4),
        ("Dec_Time_s",         "Dec Time (ms)",   1000, 4),
        ("Enc_Throughput_MBps","Enc Tput (MB/s)",    1, 3),
        ("Dec_Throughput_MBps","Dec Tput (MB/s)",    1, 3),
        ("Expansion_Ratio",    "Expansion",           1, 4),
    ]

    for algo in ALGORITHMS:
        print(f"\n{'─'*90}")
        print(f" {algo}")
        print(f"{'─'*90}")
        sub = summary_df[summary_df["Algorithm"] == algo].sort_values("File_Size_KB")

        header = f"{'Size':>9}"
        for _, label, _, _ in metrics_to_show:
            header += f"  {label:>18}"
        print(header)

        for _, r in sub.iterrows():
            kb = int(r["File_Size_KB"])
            size_str = f"{kb} KB" if kb < 1024 else f"{kb//1024} MB"
            line = f"{size_str:>9}"
            for col, _, scale, dps in metrics_to_show:
                m = r.get(f"{col}_mean", 0) * scale
                s = r.get(f"{col}_std",  0) * scale
                line += f"  {m:>9.{dps}f} ± {s:<7.{dps}f}"
            print(line)

    print("\n" + "=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Cryptographic Algorithm Performance Benchmark")
    print("  Algorithms : AES-256-GCM | RSA-2048-Hybrid | ChaCha20 | ECC-256-ECIES")
    print(f"  File sizes : {FILE_SIZES_KB} KB")
    print(f"  Runs       : {NUM_RUNS} per (algorithm × file size)")
    print("=" * 60)

    # ── Step 1: File generation ───────────────────────────────────────────────
    print("\n[1/4] Generating synthetic files …")
    file_paths = generate_synthetic_files(RANDOM_SEED)

    # ── Step 2–6: Run experiments ─────────────────────────────────────────────
    print("\n[2/4] Running encryption / decryption experiments …")
    rows = run_experiment(file_paths)

    # ── Step 7: Save raw CSV ──────────────────────────────────────────────────
    print("\n[3/4] Saving results …")
    df_raw = save_raw(rows)

    # ── Step 8: Statistics & summary CSV ─────────────────────────────────────
    print("\n[4/4] Computing statistics …")
    df_summary = compute_summary(df_raw)

    # ── Console summary table ─────────────────────────────────────────────────
    print_summary_table(df_summary)

    # ── Validation report ─────────────────────────────────────────────────────
    invalid = df_raw[df_raw["Decryption_Valid"] == False]
    if invalid.empty:
        print("\n✅  All decryptions validated — recovered data matches originals.")
    else:
        print(f"\n⚠   {len(invalid)} decryption(s) failed validation:")
        print(invalid[["Algorithm", "File_Size_KB", "Run"]].to_string(index=False))

    print(f"\nDone!  Output files:\n  {RAW_CSV}\n  {SUMMARY_CSV}\n")


if __name__ == "__main__":
    main()
