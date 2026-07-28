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
          packages = [ pkgs.python3 pkgs.uv pkgs.ruff ];

          # GPU training env: manylinux torch wheels dlopen libstdc++/zlib
          # (covered by the system nix-ld set) plus the NVIDIA driver
          # userspace libs, which live outside it at /run/opengl-driver/lib.
          shellHook = ''
            export LD_LIBRARY_PATH="''${NIX_LD_LIBRARY_PATH:+$NIX_LD_LIBRARY_PATH:}/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
