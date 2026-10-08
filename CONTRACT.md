# artifacts/results.json — the contract

The Python side writes this. The web app reads it and renders. Nothing else crosses the
boundary, so both halves can be built at the same time without touching each other.

Written to `artifacts/results.json`. Clip videos go next to it under `artifacts/clips/`.
The web app copies both into `web/public/` at build time.

```jsonc
{
  "version": "0.1",
  "generated_at": "2026-10-08T15:04:05Z",

  "clips": [
    {
      "id": "synthetic-looming",          // unique, filename-safe
      "source": "synthetic",              // synthetic | reactor | marble | veo | real
      "model": "analytic",                // e.g. "lingbot-world-2", "helios", "marble-1.1"
      "label": "looming",                 // human name for the condition
      "physical": true,                   // is this clip geometrically correct by construction?
                                          //   true  -> detectors SHOULD fire normally
                                          //   false -> this is a broken control
                                          //   null  -> unknown (all real generated clips)
      "note": "constant-velocity approach, exact ground truth",

      "fps": 60.0,
      "n_frames": 117,
      "video_url": "clips/synthetic-looming.mp4",
      "poster_url": "clips/synthetic-looming.jpg",

      // Present only when we know the real geometry (synthetic + marble).
      // null everywhere else — never fake it.
      "ground_truth": {
        "theta_deg": [5.7, 5.8, "..."],   // per frame, full angular size
        "t_contact_s": 2.0
      },

      "detectors": {
        "baseline": {
          "name": "analytic expansion",
          "fired": true,
          "fire_frame": 80,
          "fire_t_s": 1.333,
          "theta_at_fire_deg": 20.1,      // angular size when it committed
          "theta_deg": [5.6, 5.9, "..."], // per frame, as ESTIMATED from pixels
          "trace": [0.0, 0.01, "..."]     // 0..1 normalised decision variable
        },
        "fly": {
          "name": "emd + lplc2 (fallback)", // or "giant fibre (LC4 + LPLC2)"
          "fired": true,
          "fire_frame": 78,
          "fire_t_s": 1.300,
          "theta_at_fire_deg": 19.6,      // true theta at the frame it fired
          "trace": [0.0, 0.02, "..."],    // membrane potential, 0..1
          "facet_activity": [[0.1, "..."]] // (T, F) optional, decimated for the browser
        }
      },

      "scores": {
        "theta_threshold_deg": 19.6,      // the headline number
        "ttc_error_s": 0.04,              // |estimated - true| time to contact, null if unknown
        "detectors_agree": true,          // did both fire within tolerance?
        "disagreement_frames": 2          // |fly.fire_frame - baseline.fire_frame|
      }
    }
  ],

  // Facet geometry, sent once instead of per clip.
  "retina": {
    "n_facets": 223,
    "dphi_deg": 5.1,
    "drho_deg": 5.7,
    "directions": [[-40.0, 0.0], "..."]   // (F, 2) azimuth, elevation in degrees
  },

  "summary": {
    "n_clips": 12,
    "threshold_mean_deg": 19.9,
    "threshold_sd_deg": 2.7,              // invariance across conditions — lower is better
    "baseline_vs_fly_agreement": 0.83
  }
}
```

## rules

- **Never invent `ground_truth`.** It is `null` unless the geometry is known by construction
  (synthetic) or recoverable from real 3D (marble's collider mesh + metric scale).
- **`physical` is `null` for generated clips.** We do not know whether a world model's output is
  correct — that is the thing being measured. Only synthetic clips get `true`/`false`.
- Arrays are per frame and the same length as `n_frames`, except `facet_activity`, which may be
  decimated in time. If it is, it carries its own `stride`.
- Keep `results.json` under a few MB. Decimate before writing, not in the browser.
