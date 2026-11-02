# Offline 3D runtime

Three.js 0.146.0, MIT license. Vendored files are embedded into every standalone
inspector; no CDN is required at runtime. OrbitControls uses the same release.

- three.min.js: https://cdn.jsdelivr.net/npm/three@0.146.0/build/three.min.js
  SHA256 3a6602548c758cca4e2762d50e802a5400b4c663b6a638d76a9163598467ded4
- OrbitControls.js: https://cdn.jsdelivr.net/npm/three@0.146.0/examples/js/controls/OrbitControls.js
  SHA256 fab106fa8f5eafb9d16216c9b454d673fc5d974be2c20a7a6f05cb6d4e66cee2

The full MIT license is in LICENSE.three.txt and is included in generated HTML.
The version is pinned for the standalone classic-script build, without a
frontend bundler. It should be reviewed when upgrading browser/runtime support.
