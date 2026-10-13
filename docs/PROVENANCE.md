# Recorded Provenance v1

This module reads the original file with ExifTool and verifies embedded C2PA
credentials with the official Content Authenticity Initiative Python SDK.
It does not generate or change a detector score. It requires no training.

## Setup

Use Python 3.11 or newer in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-provenance.txt
```

Install [ExifTool](https://exiftool.org/install.html). On Windows, extract the
64-bit distribution, rename `exiftool(-k).exe` to `exiftool.exe`, and retain its
companion `exiftool_files` directory. Set `EXIFTOOL_PATH` to the executable's full
path, or add its directory to PATH. On Ubuntu, install `libimage-exiftool-perl`.

The pinned C2PA SDK publishes Windows x64 and Linux ARM64 wheels. Oracle's ARM
deployment still needs testing with the complete CF backend.

## Inspect an Image

Run from the repository root:

```powershell
python -m src.ai_shield_provenance "image.jpg" --output ".local/report.json"
```

On this Windows workspace an isolated portable runtime is available in the
ignored `.local/provenance/` directory. It does not install anything globally:

```powershell
./scripts/inspect_provenance.ps1 "image.jpg" -OutputPath ".local/report.json"
```

If Windows blocks unsigned local scripts, use a process-only execution policy
for this launcher; this does not change the machine's permanent policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/inspect_provenance.ps1 "image.jpg" -OutputPath ".local/report.json"
```

The launcher also works with a configured `.venv`. Use `-IncludeGps` and
`-TrustAnchors "anchors.pem"` for the corresponding optional arguments.

Add `--include-gps` to include EXIF/XMP GPS fields. They are omitted from the
metadata output by default. Other metadata and C2PA claims can also contain
personal information, so this report is for local inspection, not automatic
public sharing.

To replace the SDK trust anchors with an explicit PEM certificate bundle:

```powershell
python -m src.ai_shield_provenance "signed.jpg" --trust-anchors "anchors.pem"
```

Never trust a certificate just because it is included in an uploaded image.
Choose trust anchors independently. Development certificates do not establish
production signer trust.

## Report Contract

The report has `schema_version`, `file`, `metadata`, `provenance`, and
`interpretation` fields. Tag groups are preserved so contradictory EXIF and XMP
values remain visible. Embedded dates and this file copy's filesystem dates are
reported separately. Dates are retained as recorded, including available offset
tags; an absent timezone is not guessed.

Metadata status is `present`, `absent`, `unavailable`, or `error`. Present metadata
is unverified. `absent` means none of the requested embedded tag groups were found,
not that the file has no technical headers or is AI-generated.

C2PA status is:

| Status | Meaning |
| --- | --- |
| `validated` | SDK reports Valid or Trusted without reported validation failures |
| `untrusted` | SDK reports valid content, but signer trust could not be established |
| `failed` | SDK reports Invalid or validation failures |
| `incomplete` | Manifest read, but SDK did not establish a validation state |
| `absent` | SDK found no embedded manifest, and no extracted remote reference |
| `remote_reference` | Credentials are referenced remotely; not retrieved |
| `unsupported` | SDK cannot process this format |
| `unavailable` | SDK or its native dependency is missing |
| `error` | Reading/verification could not complete |

`signature`, `image_integrity`, and `signer_trust` are separate. A valid signature
does not establish trust in its signer, and a signature can remain valid after the
image pixels are tampered with. Detailed SDK results and claimed actions/source
types are retained even when validation fails, but those claims must not be
presented as verified in that case. `claims_binding_verified` describes the SDK's
cryptographic validation, not the factual truth of an assertion or trust in its
author. Even trusted claims do not prove scene truth.

Verification runs offline: remote manifests and live OCSP revocation responses
are not fetched. Embedded revocation information may still be processed by the
SDK. A remote reference therefore cannot be reported as verified. Reports retain
the verification policy to make these limits explicit.

## Tests

```powershell
python -m unittest discover -s tests -v
```

For real-tool tests, install Pillow (`python -m pip install "Pillow>=11,<12"`),
set `EXIFTOOL_PATH`, and set `AI_SHIELD_PROVENANCE_FIXTURES` to a directory containing
the upstream SDK fixtures `C.jpg` and `P1000827.jpg`:

- Repository: https://github.com/contentauth/c2pa-python
- Revision used locally: `192023c3edc1cd74962650e0fa0f1ec431304e9c`
- Files: `tests/fixtures/C.jpg` and
  `tests/fixtures/files-for-signing-tests/P1000827.jpg`

The integration tests inspect a camera photo, create an edited copy, create a
metadata-free JPEG, verify signed credentials, and modify signed pixels to check
that image binding fails. Upstream test certificates may not be trusted by the
SDK's default trust store; signature and binding are checked independently.

For the optional explicit-trust test, save `trust.trust_anchors` from the upstream
`tests/trust_config_test_settings.json` as `test-anchors.pem` in the fixture
directory. These are public development trust anchors, used only for testing.
The workspace's isolated tools include them; the production defaults do not.

## CF Integration Next

```python
from src.ai_shield_provenance import analyze_provenance

evidence = analyze_provenance(original_upload_path)
# Separately run frozen CF inference on this same original upload.
# Return {"detection": cf_result, "evidence": evidence} from the upload endpoint.
```

Keep the original upload intact. CF preprocessing operates on a separately
decoded image, after evidence extraction. The module accepts one local file,
limits inputs to 25 MiB, and gives ExifTool a 20-second timeout. The future API
must enforce upload limits, image-format validation, concurrency limits, and
temporary-file cleanup before public deployment. This change does not modify
the existing Streamlit or Hugging Face app.

## References

- [ExifTool JSON and grouping options](https://exiftool.org/exiftool_pod2.html)
- [C2PA SDK usage](https://github.com/contentauth/c2pa-python/blob/main/docs/usage.md)
- [C2PA verification settings](https://github.com/contentauth/c2pa-python/blob/main/docs/context-settings.md)
