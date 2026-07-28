{
  description = "balatroAI — bots for Balatro via the balatrobot JSON-RPC API";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in
    {
      packages = forAll (pkgs: rec {
        balatroai = pkgs.python3Packages.buildPythonApplication {
          pname = "balatroai";
          version = "0.1.0";
          src = ./.;
          pyproject = true;
          build-system = [ pkgs.python3Packages.hatchling ];
        };
        default = balatroai;
      });

      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = [ pkgs.python3 pkgs.uv pkgs.ruff pkgs.gcc ];

          # GPU training env, three NixOS-specific fixups:
          #  1. manylinux torch wheels dlopen libstdc++/zlib (system nix-ld
          #     set) plus the NVIDIA driver userspace libs, which live
          #     outside it at /run/opengl-driver/lib.
          #  2. triton locates libcuda by shelling out to /sbin/ldconfig,
          #     which does not exist here — point it at the driver dir.
          #  3. triton's bundled ptxas/cuobjdump/nvdisasm are prebuilt ELFs
          #     wanting /lib64/ld-linux-x86-64.so.2 (absent); re-exec them
          #     through the nix-ld glibc loader via generated wrappers.
          # torch 2.13 routes eager ops through triton, so this is not just
          # a torch.compile concern — training will not run without it.
          shellHook = ''
            export LD_LIBRARY_PATH="''${NIX_LD_LIBRARY_PATH:+$NIX_LD_LIBRARY_PATH:}/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

            # TMPDIR here is a 4 GB tmpfs; inductor/triton codegen fills it and
            # dies with ENOSPC mid-compile. Park both caches on disk, which
            # also makes compiled kernels survive across runs.
            _cache="''${XDG_CACHE_HOME:-$HOME/.cache}"
            export TORCHINDUCTOR_CACHE_DIR="$_cache/balatroai/torchinductor"
            export TRITON_CACHE_DIR="$_cache/balatroai/triton"
            mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

            # Rollout tensors and cuda-graph pools are long-lived while the
            # per-minibatch activations churn; expandable segments keep that
            # mix from fragmenting a 24 GB card into an OOM.
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

            _tb=$(echo "$PWD"/.venv/lib/python*/site-packages/triton/backends/nvidia/bin)
            if [ -d "$_tb" ] && [ -n "$NIX_LD" ]; then
              _wrap="$PWD/.venv/nix-ld-bin"
              mkdir -p "$_wrap"
              for _b in ptxas cuobjdump nvdisasm; do
                [ -f "$_tb/$_b" ] || continue
                printf '#!/bin/sh\nexec "%s" "%s" "$@"\n' "$NIX_LD" "$_tb/$_b" > "$_wrap/$_b"
                chmod +x "$_wrap/$_b"
              done
              export TRITON_PTXAS_PATH="$_wrap/ptxas"
              export TRITON_CUOBJDUMP_PATH="$_wrap/cuobjdump"
              export TRITON_NVDISASM_PATH="$_wrap/nvdisasm"
            fi
            unset _tb _wrap _b
          '';
        };
      });

      # Doctrine 06: the bare core (poker eval + CLI + random bot import path)
      # must boot with zero policy/config and no game server.
      checks = forAll (pkgs: {
        bare-core = pkgs.runCommand "balatroai-bare-core" { } ''
          export PYTHONPATH=${./src}
          ${pkgs.python3}/bin/python -m balatroai.poker
          ${pkgs.python3}/bin/python -m balatroai.cli --help > /dev/null
          ${pkgs.python3}/bin/python -c "from balatroai.bots import get_bot; get_bot('random')"
          touch $out
        '';
      });
    };
}
