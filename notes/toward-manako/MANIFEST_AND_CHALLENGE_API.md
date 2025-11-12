# Manifest Distribution & Challenge API Changes

## Manifest Distribution Architecture

### Who Publishes the Manifest?

**Protocol Operators (Score)** publish the Manifest at the start of each evaluation window. This is currently centralized governance (v1.3), but the Manifest itself is publicly verifiable and content-addressed.

### How Does Distribution Work?

#### 1. Manifest Publication Flow

```
Protocol Operator (Score)
  ↓
1. Create Manifest JSON with all parameters
2. Sign Manifest with Ed25519 private key
3. Compute SHA-256 hash of signed Manifest → manifest_hash
4. Upload to R2/CDN at: scorevision/manifests/{manifest_hash}.json
5. Update index: scorevision/index.json → add manifest_hash
6. Publish manifest_hash on-chain (using ownerkey?)
```

#### 2. Manifest Fetching (Everyone Can Get It)

**Validators and Miners** fetch the Manifest independently:

```python
# Public URL pattern
manifest_url = f"{R2_PUBLIC_URL}/scorevision/manifests/{manifest_hash}.json"

# Or via index lookup
index_url = f"{R2_PUBLIC_URL}/scorevision/index.json"
# Returns: {"manifests": [{"hash": "...", "window_id": "...", "expiry_block": ...}]}
```

**Key Points:**

- ✅ **Public Access**: No authentication required to read Manifest
- ✅ **Content-Addressed**: SHA-256 hash ensures integrity
- ✅ **Verifiable**: Signature proves authenticity
- ✅ **Cached**: CDN caching for fast global access
- ✅ **Immutable**: Once published, Manifest cannot change (new hash = new Manifest)

#### 3. Manifest Discovery

Two methods for finding the current Manifest:

**Method A: Direct Hash (if known)**

```python
manifest_hash = "abc123..."  # From on-chain commitment or index
manifest = await fetch_manifest(manifest_hash)
```

**Method B: Index Lookup (for current window)**

```python
index = await fetch_index()  # scorevision/index.json
current_block = await get_current_block()
current_manifest = None
for m in index["manifests"]:
    if m["expiry_block"] > current_block:
        current_manifest = await fetch_manifest(m["hash"])
        break
```

---

### Miner Element Catalog & Upcoming Windows

Miners consume the same manifest index to understand which Elements are open for the **current** and **upcoming** windows before committing resources:

1. `sv miner elements --window current`  
   - Reads `scorevision/index.json`, fetches the active manifest, and prints each Element's `element_id`, service rate, θ, β, clip counts, and expected telemetry.
2. `sv miner elements --window upcoming`  
   - Looks at the next manifest entry (highest expiry_block greater than the active one) so miners can prep models before the window flips.
3. `sv miner manifest --hash sha256:...`  
   - Downloads and verifies any manifest locally for offline planning.

These commands are backed by `scorevision/utils/element_catalog.py`, which caches manifests locally and exposes helper methods such as `list_elements(window_scope)` and `summarize_window(window_id)`.

Once a miner knows which Elements they can realistically support, they submit an on-chain commitment declaring the `(window_id, element_ids)` pairs they plan to serve. Validators later reference these commitments when accepting shards, creating a shared source of truth between manifest distribution, API usage, and miner economics.

### On-Chain Element Commitments

**Purpose:** Commitments provide an auditable declaration of the Elements a miner will serve for a specific evaluation window, preventing opportunistic cherry-picking mid-window and aligning Challenge API requests with on-chain state.

**CLI Flow:**

```bash
# Declare participation for the next window
sv miner commit \
  --window-id 2025-10-23 \
  --elements PlayerDetect_v1@1.0,BallDetect_v1@1.0 \
  --service-cap-fps 45

# Withdraw (allowed only before window start)
sv miner commit --withdraw --window-id 2025-10-23 --elements PlayerDetect_v1@1.0
```

Under the hood, `scorevision/utils/commitments.py`:

1. Loads the miner hotkey (same key used for challenge requests)
2. Signs and submits `commit_element_set(window_id, element_ids, service_cap_fps)` against Bittensor
3. Waits for inclusion and captures `commitment_proof = f"{block_number}:{extrinsic_hash}"`
4. Stores the proof locally for reuse across API calls and shard telemetry

**State Layout (conceptual):**

```json
{
  "miner_hotkey": "ss58:abc...",
  "window_id": "2025-10-23",
  "element_ids": ["PlayerDetect_v1@1.0", "BallDetect_v1@1.0"],
  "bindings": [
    {
      "element_id": "PlayerDetect_v1@1.0",
      "hf_revision": "player-v1",
      "chute_slug": "scorevision-turbovision-hfuser-lynx",
      "chute_id": "87f7d0c5-...",
      "service_cap_fps": 25
    },
    {
      "element_id": "BallDetect_v1@1.0",
      "hf_revision": "ball-v1",
      "chute_slug": "scorevision-turbovision-hfuser-orca",
      "chute_id": "63e8aa01-...",
      "service_cap_fps": 20
    }
  ],
  "service_cap_fps": 45,
  "committed_at_block": 123400,
  "expires_at_block": 123456
}
```

**Validator / Backend Usage:**

- Challenge API double-checks that `(hotkey, element_id, window_id)` exists before issuing work
- Challenge API also validates that the `chute_slug` in the commitment matches the slug runners use when calling the element’s miner chute
- Validators require shards to quote the same `commitment_proof` and slug; mismatches trigger rejection
- Upcoming window commitments are encouraged so miners don't miss the first clips after rotation

---

## Challenge API Changes

### Current API (Legacy - Will Remain)

**Endpoint:** `GET {SCOREVISION_API}/api/tasks/next/v2`

**Request:**

```python
# Query params with Ed25519 signature
params = build_validator_query_params(keypair)
GET /api/tasks/next/v2?{params}
```

**Response:**

```json
{
  "task_id": "...",
  "video_url": "https://...",
  "fps": 25,
  "challenge_type": "football",
  "seed": 12345
}
```

**Status:** Legacy endpoint, will remain for backward compatibility during migration.

### New API (Manifest-Aware)

**Endpoint:** `GET {SCOREVISION_API}/api/challenge/v3`

**Request Headers:**

```python
headers = {
    "X-Manifest-Hash": "sha256:abc123...",  # REQUIRED: Current manifest hash
    "X-Commitment-Proof": "123450:0xdeadbeef",  # REQUIRED: Block:extrinsic hash proving commitment
    "X-Client-Id": "uid_123",                # Optional: For rate limiting
    "X-Nonce": "uuid-v4",                     # Optional: Prevent replay attacks
    "X-Signature": "ed25519:...",            # Optional: Ed25519 signature of request
}
```

Requests missing a valid `X-Commitment-Proof` receive `412 Precondition Failed`. Backend pulls the referenced extrinsic, ensures the miner hotkey, window, and element match the payload, and that the commitment has not expired.

**Request Query Params:**

```python
params = {
    **build_validator_query_params(keypair),  # Includes miner hotkey + signature
    "element_id": "PlayerDetect_v1@1.0",      # REQUIRED: Only request Elements you committed to
}
```

**Response (New Schema):**

```json
{
  "clip_url": "https://cdn.../video.mp4",
  "element_id": "PlayerDetect_v1@1.0",
  "window_id": "2025-10-23",
  "meta": {
    "duration_ms": 30000,
    "fps": 25,
    "resolution": {"width": 1920, "height": 1080},
    "sport": "football",
    "scenario": "match",
    "challenge_type": "football"
  },
  "manifest_hash": "sha256:abc123...",  # Echo back for verification
  "expiry_block": 123456                 # Window expiry block
}
```

**Error Responses:**

```json
// 401 Unauthorized - Invalid signature
{
  "error": "invalid_signature",
  "message": "Ed25519 signature verification failed"
}

// 404 Not Found - No active window
{
  "error": "no_active_window",
  "message": "No evaluation window active for manifest_hash"
}

// 409 Conflict - Rate limit exceeded
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Retry after: 60s",
  "retry_after": 60
}

// 410 Gone - Window expired
{
  "error": "window_expired",
  "message": "Evaluation window expired at block 123456",
  "expired_at_block": 123456,
  "current_block": 123500
}

// 412 Precondition Failed - Missing/invalid commitment
{
  "error": "missing_commitment",
  "message": "Miner has no commitment for element PlayerDetect_v1@1.0 in window 2025-10-23",
  "hotkey": "ss58:abc...",
  "window_id": "2025-10-23"
}
```

---

## Implementation Details

### 1. Manifest Storage Structure

**R2/CDN Path:**

```
scorevision/
  ├── index.json                    # Current manifests index
  ├── manifests/
  │   ├── {manifest_hash_1}.json   # Signed Manifest 1
  │   ├── {manifest_hash_2}.json   # Signed Manifest 2
  │   └── ...
  └── ...
```

**index.json Format:**

```json
{
  "version": "1.0",
  "manifests": [
    {
      "hash": "sha256:abc123...",
      "window_id": "2025-10-23",
      "expiry_block": 123456,
      "published_at": "2025-10-23T00:00:00Z",
      "published_by": "ss58:...",
      "elements_summary": [
        {
          "id": "PlayerDetect_v1@1.0",
          "beta": 1.0,
          "baseline_theta": 0.78,
          "service_rate_fps": 25
        },
        {
          "id": "BallDetect_v1@1.0",
          "beta": 1.4,
          "baseline_theta": 0.70,
          "service_rate_fps": 30
        }
      ]
    }
  ],
  "current_manifest_hash": "sha256:abc123..."
}
```

### 2. Manifest Fetching Implementation

**New Module: `scorevision/utils/manifest.py`**

```python
async def fetch_manifest(manifest_hash: str) -> Manifest:
    """Fetch Manifest from R2/CDN by hash."""
    url = f"{R2_PUBLIC_URL}/scorevision/manifests/{manifest_hash}.json"
    async with get_async_client() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return Manifest.parse_obj(data)

async def get_current_manifest(block_number: int | None = None) -> Manifest | None:
    """Get the active Manifest for current block."""
    if block_number is None:
        st = await get_subtensor()
        block_number = await st.get_current_block()

    index = await fetch_index()
    for m in index["manifests"]:
        if m["expiry_block"] > block_number:
            return await fetch_manifest(m["hash"])
    return None

async def fetch_index() -> dict:
    """Fetch the manifest index."""
    url = f"{R2_PUBLIC_URL}/scorevision/index.json"
    async with get_async_client() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()

def verify_manifest_signature(manifest: Manifest) -> bool:
    """Verify Manifest Ed25519 signature."""
    # Implementation using nacl.signing
    ...
```

### 3. Challenge API Client Changes

**Update: `scorevision/utils/challenges.py`**

```python
async def get_next_challenge_v3(
    manifest_hash: str | None = None,
    element_id: str | None = None,
) -> dict:
    """
    Fetches challenge from new Manifest-aware Challenge API v3.

    Args:
        manifest_hash: Current manifest hash (required for new API)
        element_id: Specific Element to request (must match commitment)

    Returns:
        Challenge dict with clip_url, element_id, window_id, meta
    """
    settings = get_settings()

    # Get current manifest if not provided
    if manifest_hash is None:
        manifest = await get_current_manifest()
        if manifest is None:
            raise ScoreVisionChallengeError("No active manifest found")
        manifest_hash = manifest.hash

    keypair = load_hotkey_keypair(...)
    hotkey = keypair.ss58_address

    # Ensure we know which Element we're requesting
    if element_id is None:
        element_id = select_element_from_commitments(window_id=window_id)
        if element_id is None:
            raise ScoreVisionChallengeError("No committed Elements for current window")

    commitment = await get_commitment_proof(
        hotkey=hotkey,
        window_id=window_id,
        element_id=element_id,
    )
    slug = commitment.chute_slug
    revision = commitment.hf_revision

    # Build headers with Manifest hash
    headers = {
        "X-Manifest-Hash": manifest_hash,
        "X-Commitment-Proof": commitment.proof,  # "{block}:{extrinsic_hash}"
    }

    # Add signature (Ed25519)
    nonce = str(uuid.uuid4())
    payload = f"{manifest_hash}:{nonce}"
    signature = keypair.sign(payload.encode()).hex()
    headers["X-Nonce"] = nonce
    headers["X-Signature"] = f"ed25519:{signature}"

    session = await get_async_client()
    params = {
        **build_validator_query_params(keypair),
        "element_id": element_id,
    }

    async with session.get(
        f"{settings.SCOREVISION_API}/api/challenge/v3",  # NEW ENDPOINT
        headers=headers,
        params=params,
    ) as response:
        if response.status == 404:
            raise ScoreVisionChallengeError("No active window")
        if response.status == 409:
            retry_after = int(response.headers.get("Retry-After", 60))
            raise ScoreVisionChallengeError(f"Rate limited. Retry after {retry_after}s")
        if response.status == 410:
            raise ScoreVisionChallengeError("Window expired")
        if response.status == 401:
            raise ScoreVisionChallengeError("Invalid signature")

        response.raise_for_status()
        challenge = await response.json()

        # Validate response includes required fields
        if not challenge.get("clip_url"):
            raise ScoreVisionChallengeError("Challenge missing clip_url")
        if not challenge.get("element_id"):
            raise ScoreVisionChallengeError("Challenge missing element_id")
        if not challenge.get("window_id"):
            raise ScoreVisionChallengeError("Challenge missing window_id")

        # Verify manifest_hash matches
        if challenge.get("manifest_hash") != manifest_hash:
            logger.warning(
                f"Manifest hash mismatch: expected {manifest_hash}, "
                f"got {challenge.get('manifest_hash')}"
            )

        challenge["element_id"] = element_id
        challenge["chute_slug"] = slug
        challenge["hf_revision"] = revision
        return challenge
```

`select_element_from_commitments()` reads cached receipts (written by `sv miner commit`) and picks either the CLI-specified element or the next in rotation. `get_commitment_proof()` fetches the `{block}:{extrinsic_hash}` tuple plus the stored `hf_revision/chute_slug/chute_id` mapping from the same receipt set and refreshes it via RPC if it has expired. The runner then uses `chute_slug` when calling `call_miner_model_on_chutes()`, guaranteeing that each Element hits the chute the miner declared for that Element.

### 4. Runner Integration

**Update: `scorevision/cli/runner.py`**

```python
async def runner(slug: str | None = None) -> None:
    # ... existing setup ...

    # Check which API endpoint to use (config/env variable)
    use_v3_api = os.getenv("SCOREVISION_USE_CHALLENGE_V3", "0") in ("1", "true", "True")

    if use_v3_api:
        # Fetch current manifest
        manifest = await get_current_manifest()
        if manifest is None:
            logger.error("No active manifest found")
            return

        manifest_hash = manifest.hash
        window_id = manifest.window_id

        # Choose which Element to request based on on-chain commitments
        element_id = select_element_from_commitments(window_id=window_id)
        if element_id is None:
            logger.error("No element commitments found for this window")
            return

        # Get challenge from new v3 API with manifest hash + commitment proof
        challenge_dict = await get_next_challenge_v3(
            manifest_hash=manifest_hash,
            element_id=element_id,
        )

        # Verify window_id matches
        if challenge_dict.get("window_id") != window_id:
            logger.error(f"Window ID mismatch: {window_id} != {challenge_dict.get('window_id')}")
            return

        # Sanity check: element returned should match what we requested/committed
        returned_element = challenge_dict.get("element_id")
        if returned_element != element_id:
            logger.error(f"Element mismatch: requested {element_id}, got {returned_element}")
            return
    else:
        # Legacy flow - use existing endpoint
        challenge_dict = await get_next_challenge()  # Existing function
        element_id = None  # Not available in legacy API

    # ... rest of runner logic ...
    # Include manifest_hash, window_id, element_id, chute_slug, and commitment_proof in shards (if using v3)
```

### 5. Challenge API Server Changes (Backend)

**Required Backend Changes:**

1. **New Endpoint:** `GET /api/challenge/v3` (completely separate from existing)

   - Accept `X-Manifest-Hash` header (required)
   - Accept `X-Commitment-Proof` header (required) and verify the referenced extrinsic binds `(hotkey, window_id, element_id, chute_slug)`
   - Validate manifest_hash exists and is active
   - Check window expiry (return 410 if expired)
   - Ensure requested `element_id` appears in both the manifest and the on-chain commitment and return the stored slug/revision to the runner
   - Implement rate limiting (return 409 if exceeded)
   - Return `element_id` and `window_id` in response
   - **No changes to existing `/api/tasks/next/v2` endpoint**

2. **Response Schema:**

   ```python
   {
       "clip_url": challenge.video_url,
       "element_id": determine_element_id(challenge),  # From challenge metadata
       "window_id": get_window_id_for_manifest(manifest_hash),
       "meta": {
           "duration_ms": challenge.duration_ms,
           "fps": challenge.fps,
           "resolution": challenge.resolution,
           "sport": challenge.sport,
           "scenario": challenge.scenario,
       },
       "manifest_hash": manifest_hash,
       "expiry_block": manifest.expiry_block,
   }
   ```

3. **Window Management:**

   - Track active windows per manifest_hash
   - Return 404 if no window active for manifest_hash
   - Return 410 if window expired

4. **Migration Path:**
   - New endpoint runs in parallel with existing
   - Runners switch via config/env variable
   - Old endpoint remains unchanged (no risk to existing infrastructure)
   - Can switch back if needed by changing config

---

## Migration Strategy

### Phase 1: New Endpoint Deployment (Week 1)

- ✅ Implement new `/api/challenge/v3` endpoint (parallel to existing)
- ✅ Keep `/api/tasks/next/v2` unchanged (no modifications)
- ✅ Update runner to use new endpoint via config/env variable
- ✅ Test new endpoint independently

### Phase 2: Manifest Integration (Week 2)

- ✅ Publish first Manifest to R2
- ✅ New endpoint reads from Manifest
- ✅ Return `element_id` and `window_id` in responses
- ✅ Validate `X-Manifest-Hash` header
- ✅ Enforce `X-Commitment-Proof` header + element commitments for miners
- ✅ Switch runners to new endpoint via config

### Phase 3: Full Migration (Week 3+)

- ✅ All validators using `/api/challenge/v3`
- ✅ Monitor for any issues
- ✅ Eventually deprecate `/api/tasks/next/v2` (future)

**Key Benefit:** Clean separation - new endpoint doesn't affect existing infrastructure. Switch over via configuration when ready.

---

## Key Benefits

1. **Version Consistency**: Manifest hash ensures all participants use same rules
2. **Window Awareness**: Explicit window_id prevents cross-window confusion
3. **Element Clarity**: element_id makes challenge purpose explicit
4. **Miner Commitments**: Api enforces `X-Commitment-Proof`, so only pre-declared Elements receive work
5. **Public Verification**: Anyone can fetch and verify Manifest
6. **Zero-Risk Migration**: New endpoint doesn't affect existing infrastructure
7. **Easy Rollback**: Switch back to legacy endpoint via config if needed
8. **Clean Separation**: No need to maintain backward compatibility in same endpoint

---

## Testing Checklist

- [ ] Manifest can be published to R2 and fetched publicly
- [ ] Manifest signature verification works
- [ ] Miner CLI lists current + upcoming Elements directly from manifest index
- [ ] `sv miner commit` submits on-chain commitments and caches proof locally
- [ ] Commitment payload stores `hf_revision`, `chute_slug`, and `chute_id` per Element and backend returns them when challenges are requested
- [ ] New `/api/challenge/v3` endpoint accepts `X-Manifest-Hash` header
- [ ] New endpoint enforces `X-Commitment-Proof` and returns 412 when missing/invalid
- [ ] New endpoint returns `element_id` and `window_id`
- [ ] New endpoint returns 410 for expired windows
- [ ] New endpoint returns 404 for inactive windows
- [ ] New endpoint returns 409 for rate limits
- [ ] Legacy `/api/tasks/next/v2` endpoint still works unchanged
- [ ] Runner can switch between endpoints via config
- [ ] Runner fetches Manifest before getting challenges (v2 mode)
- [ ] Runner passes `element_id` + `X-Commitment-Proof` into challenge requests and shards
- [ ] Runner includes manifest_hash in shards (v2 mode)
- [ ] Validator can verify manifest_hash in shards
- [ ] Can rollback to legacy endpoint if needed
